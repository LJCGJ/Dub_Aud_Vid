// Metadados de versão no dub.exe (Windows): nome do produto, versão, descrição.
// Exigido pela política de assinatura (SignPath) e útil no Gerenciador de Tarefas.
fn main() {
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set("ProductName", "Dub");
        res.set("FileDescription", "Dub — dublagem automática de vídeos (CLI)");
        res.set("CompanyName", "Leonardo Gonzaga");
        res.set("LegalCopyright", "MIT License");
        res.set("OriginalFilename", "dub.exe");
        res.set("InternalName", "dub");
        res.set_icon("../../app/icons/icon.ico");
        res.compile().expect("recursos do Windows");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
