use crate::settings::Settings;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

pub struct RunningJob {
    pub cancel: Arc<AtomicBool>,
}

pub struct AppState {
    pub settings: Mutex<Settings>,
    pub job: Mutex<Option<RunningJob>>,
    pub downloading: AtomicBool,
}

impl AppState {
    pub fn new(settings: Settings) -> Self {
        Self { settings: Mutex::new(settings), job: Mutex::new(None), downloading: AtomicBool::new(false) }
    }
}
