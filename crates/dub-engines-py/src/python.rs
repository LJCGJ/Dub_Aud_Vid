use dub_core::{DubError, Result};
use std::path::{Path, PathBuf};

/// Localiza o interpretador Python e a pasta do pacote `dub_engines`.
///
/// Ordem de busca do interpretador:
/// 1. `DUB_PYTHON` (variável de ambiente)
/// 2. venv do projeto em `py/.venv`
/// 3. `python3` / `python` no PATH
#[derive(Debug, Clone)]
pub struct PythonEnv {
    pub interpreter: PathBuf,
    /// Diretório que contém `dub_engines/` (vai para `PYTHONPATH`).
    pub package_dir: PathBuf,
}

impl PythonEnv {
    pub fn discover(project_root: Option<&Path>) -> Result<Self> {
        let root = project_root.map(Path::to_path_buf).or_else(find_project_root);
        let package_dir = std::env::var_os("DUB_ENGINES_DIR")
            .map(PathBuf::from)
            .or_else(|| root.as_ref().map(|r| r.join("py")))
            .ok_or_else(|| DubError::ToolMissing("pasta py/ com dub_engines (defina DUB_ENGINES_DIR)".into()))?;

        if let Some(p) = std::env::var_os("DUB_PYTHON") {
            return Ok(Self { interpreter: PathBuf::from(p), package_dir });
        }
        let venv_bin = if cfg!(windows) { package_dir.join(".venv/Scripts/python.exe") } else { package_dir.join(".venv/bin/python") };
        if venv_bin.exists() {
            return Ok(Self { interpreter: venv_bin, package_dir });
        }
        for name in ["python3", "python"] {
            if let Ok(p) = which::which(name) {
                return Ok(Self { interpreter: p, package_dir });
            }
        }
        Err(DubError::ToolMissing("python (defina DUB_PYTHON ou crie py/.venv)".into()))
    }
}

/// Sobe diretórios a partir do executável e do cwd procurando `py/dub_engines`.
fn find_project_root() -> Option<PathBuf> {
    let mut candidates = vec![std::env::current_dir().ok()];
    if let Ok(exe) = std::env::current_exe() {
        candidates.push(exe.parent().map(Path::to_path_buf));
    }
    for start in candidates.into_iter().flatten() {
        let mut dir = Some(start.as_path());
        while let Some(d) = dir {
            if d.join("py").join("dub_engines").is_dir() {
                return Some(d.to_path_buf());
            }
            dir = d.parent();
        }
    }
    None
}
