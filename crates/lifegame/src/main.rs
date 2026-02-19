use lifegame::App;
use winit::event_loop::EventLoop;

fn main() {
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();
    let event_loop = EventLoop::with_user_event().build().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
