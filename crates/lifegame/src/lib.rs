use bytemuck::{Pod, Zeroable};
use encase::ShaderSize;
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use wgpu::{util::DeviceExt, SurfaceError};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, encase::ShaderType)]
struct Params {
    world_size: glam::u32::UVec2,
    // https://conwaylife.com/wiki/Rule_integer
    rule: u32,
}

// https://github.com/gfx-rs/wgpu/blob/v27/examples/features/src/boids/mod.rs
pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    compute_buffers: Vec<wgpu::Buffer>,
    compute_bind_groups: Vec<wgpu::BindGroup>,
    render_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    frame_num: usize,
    current_buffer_index: usize,
    params: Params,
    window: Arc<Window>,
    update_delay: Arc<AtomicU64>,
    update_loop_handle: std::thread::JoinHandle<()>,
    mouse_pos_x: u32,
    mouse_pos_y: u32,
}

const PARKING_FLAG: u64 = 1 << (u64::BITS - 1);

impl State {
    pub async fn new(window: Arc<Window>) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;
        let surface_caps = surface.get_capabilities(&adapter);

        log::info!("surface cap: {:#?}", surface_caps);

        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![surface_format],
            desired_maximum_frame_latency: 2,
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        surface.configure(&device, &config);
        let compute_shader = device.create_shader_module(wgpu::include_wgsl!("compute.wgsl"));
        let draw_shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        // buffer for simulation parameters uniform

        let params: Params = Params {
            world_size: glam::UVec2::new(64, 64),
            rule: (1 << 3) | (((1 << 2) | (1 << 3)) << 8),
            // rule: 0xFF_00,
        };

        let mut buf = encase::UniformBuffer::new(Vec::<u8>::new());
        buf.write(&params)?;
        let buf = buf.into_inner();

        let sim_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Parameter Buffer"),
            contents: &buf,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // create compute bind layout group and compute pipeline layout

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(Params::SHADER_SIZE.into()),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (4 * params.world_size.x * params.world_size.y) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (4 * params.world_size.x * params.world_size.y) as _,
                            ),
                        },
                        count: None,
                    },
                ],
                label: None,
            });

        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Params::SHADER_SIZE.into(),
                    },
                    count: None,
                }],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut compute_buffers = Vec::<wgpu::Buffer>::new();
        let mut compute_bind_groups = Vec::<wgpu::BindGroup>::new();

        for _ in 0..2 {
            let buffer = device.create_buffer(&wgpu::wgt::BufferDescriptor {
                label: Some("compute buffer"),
                size: (4 * params.world_size.x * params.world_size.y) as _,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            });

            compute_buffers.push(buffer);
        }

        // create two bind groups, one for each buffer as the src
        // where the alternate buffer is used as the dst

        for i in 0..2 {
            compute_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sim_param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: compute_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: compute_buffers[(i + 1) % 2].as_entire_binding(),
                    },
                ],
                label: None,
            }));
        }

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render bind group"),
            layout: &render_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sim_param_buffer.as_entire_binding(),
            }],
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &draw_shader,
                entry_point: Some("main_vs"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<u32>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Uint32],
                    },
                    Vertex::desc(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &draw_shader,
                entry_point: Some("main_fs"),
                compilation_options: Default::default(),
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        let update_delay = Arc::new(AtomicU64::new(50));
        let update_loop_handle = std::thread::spawn({
            let update_delay = Arc::clone(&update_delay);
            let window = Arc::clone(&window);
            move || loop {
                let delay = update_delay.load(std::sync::atomic::Ordering::Acquire);
                if delay == 0 {
                    break;
                }
                std::thread::sleep(Duration::from_millis(delay & !PARKING_FLAG));
                if (delay & PARKING_FLAG) != 0 {
                    continue;
                }

                window.request_redraw();
            }
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            compute_pipeline,
            compute_bind_groups,
            compute_buffers,
            render_bind_group,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            frame_num: 0,
            current_buffer_index: 0,
            window,
            params,
            update_delay,
            update_loop_handle,
            mouse_pos_x: 0,
            mouse_pos_y: 0,
        })
    }

    fn need_update(&self) -> bool {
        (self.update_delay.load(Ordering::Relaxed) & PARKING_FLAG) == 0
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }

        self.window.request_redraw();
    }

    pub fn update(&mut self) -> Result<(), SurfaceError> {
        println!("update");
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        command_encoder.push_debug_group("compute");
        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_groups[self.current_buffer_index], &[]);
            cpass.dispatch_workgroups(self.params.world_size.x, self.params.world_size.y, 1);
        }
        command_encoder.pop_debug_group();
        self.queue.submit([command_encoder.finish()]);

        self.current_buffer_index = (self.current_buffer_index + 1) % 2;

        Ok(())
    }

    pub fn render(&mut self) -> Result<(), SurfaceError> {
        println!("before render");
        if !self.is_surface_configured {
            return Ok(());
        }

        if self.need_update() {
            self.update()?;
        }

        let output = self.surface.get_current_texture()?;

        let view = output.texture.create_view(&Default::default());

        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })];
        let render_pass_descriptor = wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        };

        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        command_encoder.push_debug_group("render");
        {
            // render pass
            let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.render_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.compute_buffers[self.current_buffer_index].slice(..));
            rpass.set_vertex_buffer(1, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(
                0..self.num_indices,
                0,
                0..(self.params.world_size.x * self.params.world_size.y),
            );
        }
        command_encoder.pop_debug_group();

        self.frame_num += 1;

        self.queue.submit([command_encoder.finish()]);
        output.present();

        println!("after render");
        Ok(())
    }
}

pub struct App {
    state: Option<State>,
}

impl App {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        println!("resumed");
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        if let Some(state) = self.state.take() {
            state.update_delay.store(0, Ordering::Release);
            state.update_loop_handle.join().unwrap();
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        self.state = Some(pollster::block_on(State::new(window)).unwrap());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(v) => v,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => match state.render() {
                Ok(_) => (),
                Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                    let size = state.window.inner_size();
                    state.resize(size.width, size.height);
                }
                Err(e) => {
                    log::error!("unable to render {}", e)
                }
            },
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };

                match code {
                    KeyCode::Space => {
                        let old = state
                            .update_delay
                            .fetch_xor(PARKING_FLAG, Ordering::Release);

                        let paused = (old & PARKING_FLAG) == 0;

                        let title = if paused { "paused" } else { "running" };
                        state.window.set_title(title);
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseInput {
                state: s, button, ..
            } if s.is_pressed() => match button {
                MouseButton::Left => {
                    let delay = state.update_delay.load(Ordering::Acquire);
                    let paused = (delay & PARKING_FLAG) != 0;
                    println!("pressed {:?}", button);
                    if !paused {
                        return;
                    }

                    let queue = &state.queue;

                    let buffer = &state.compute_buffers[state.current_buffer_index];

                    let size = state.window.inner_size();
                    let world_size = state.params.world_size;

                    let cell_w = size.width as f64 / world_size.x as f64;
                    let cell_h = size.height as f64 / world_size.y as f64;

                    let cell_x = (state.mouse_pos_x as f64 / cell_w) as u32;
                    let cell_y = world_size.y - (state.mouse_pos_y as f64 / cell_h) as u32;
                    let offset = (cell_x + cell_y * world_size.x) as u64 * 4;

                    println!("update cell ({}, {})", cell_x, cell_y);

                    queue.write_buffer(buffer, offset, bytemuck::bytes_of(&1));
                    queue.submit([]);

                    state.window.request_redraw();
                }
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                let posx = position.x as u32;
                let posy = position.y as u32;

                state.mouse_pos_x = posx;
                state.mouse_pos_y = posy;
            }

            _ => {}
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
}
impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            }],
        }
    }
}
/// (0, 0)       (1, 0)
///   A ---------- C
///   |            |
///   B ---------- D
/// (0, 1)       (1, 1)
const VERTICES: &[Vertex] = &[
    Vertex {
        position: [0.0, 0.0], // A
    },
    Vertex {
        position: [0.0, 1.0], // B
    },
    Vertex {
        position: [1.0, 0.0], // C
    },
    Vertex {
        position: [1.0, 1.0], // D
    },
];

const INDICES: &[u16] = &[
    0, 1, 2, // A B C
    1, 3, 2, // B D C
];
