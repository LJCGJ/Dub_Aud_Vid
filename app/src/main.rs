//! Interface de desktop do dub (Tauri 2). Frontend em HTML/CSS/JS puro em `ui/`;
//! toda a lógica pesada vive no `dub-runtime`.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod settings;
mod state;

use tauri::Manager;
use tracing_subscriber::EnvFilter;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn,dub=info")))
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let settings = settings::load(app.handle());
            app.manage(state::AppState::new(settings));
            // Recursos empacotados (onnxruntime, espeak-ng-data, models/) ficam na
            // pasta de recursos do bundle; apontamos as variáveis que o dub-native lê.
            if let Ok(res) = app.path().resource_dir() {
                settings::export_resource_env(&res);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::doctor,
            commands::list_models,
            commands::download_models,
            commands::start_dub,
            commands::cancel_dub,
            commands::job_running,
            commands::get_settings,
            commands::save_settings,
            commands::pick_video,
            commands::pick_folder,
            commands::open_path,
            commands::reveal_path,
            commands::voices_available,
        ])
        .run(tauri::generate_context!())
        .expect("erro ao iniciar o app");
}
