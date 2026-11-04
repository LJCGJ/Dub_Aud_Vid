# Assinatura de código (SignPath Foundation)

Este guia é o passo a passo para colocar a assinatura digital nos binários do
Windows usando o programa gratuito para open source do SignPath. O workflow de
release já está preparado (`.github/workflows/release.yml`): ele fica inativo
até as variáveis abaixo existirem, e passa a assinar sozinho depois.

## 1. Pré-requisitos (uma vez)

- **2FA ativado na sua conta GitHub** (Settings → Password and authentication).
  É exigência do programa para todos os membros da equipe.
- O projeto já ter uma release publicada no formato que será assinado ✔ (v0.1.0).
- A seção *Política de assinatura de código* no README ✔.

## 2. Pedido ao SignPath Foundation

1. Acesse https://signpath.org/apply e preencha:
   - **Project**: `Dub_Aud_Vid` — repositório `https://github.com/LJCGJ/Dub_Aud_Vid`
   - **License**: MIT
   - **Description**: *Local, offline automatic video dubbing (English → Brazilian
     Portuguese) for desktop: whisper.cpp transcription, CTranslate2 translation,
     Piper TTS. Rust + C++ with a Tauri UI. Ships a Windows NSIS installer.*
   - **Artifacts to sign**: `dub-app.exe` (desktop app), `dub.exe` (CLI) e o
     instalador NSIS `Dub_<versão>_x64-setup.exe`.
   - **Third-party binaries included (unsigned)**: ffmpeg/ffprobe (GPL, BtbN
     builds), yt-dlp (Unlicense), onnxruntime (MIT, Microsoft), espeak-ng-data (GPL).
   - **Build system**: GitHub Actions, workflow `release.yml`, disparado por tag `v*`.
   - Use um e-mail que você lê; a resposta costuma levar alguns dias.
2. Ao ser aprovado, você recebe acesso a uma organização no portal
   https://app.signpath.io com um certificado emitido para *SignPath Foundation*.

## 3. Configuração no portal SignPath

### 3.1 Projeto
- **Projects → Add project**: nome `Dub`, slug `dub`.
- **Repository URL**: `https://github.com/LJCGJ/Dub_Aud_Vid`.

### 3.2 Artifact configurations (duas)

**Slug `executables`** — o zip com os dois executáveis (estágio 1):

```xml
<?xml version="1.0" encoding="utf-8"?>
<artifact-configuration xmlns="http://signpath.io/artifact-configuration/v1">
  <zip-file>
    <pe-file path="dub-app.exe">
      <authenticode-sign />
    </pe-file>
    <pe-file path="dub.exe">
      <authenticode-sign />
    </pe-file>
  </zip-file>
</artifact-configuration>
```

**Slug `installer`** — o instalador NSIS (estágio 2; os executáveis dentro dele
já foram assinados no estágio 1):

```xml
<?xml version="1.0" encoding="utf-8"?>
<artifact-configuration xmlns="http://signpath.io/artifact-configuration/v1">
  <zip-file>
    <pe-file path="*-setup.exe">
      <authenticode-sign />
    </pe-file>
  </zip-file>
</artifact-configuration>
```

Em cada uma, ative as **metadata restrictions**: `product-name = Dub` e
`product-version` igual à versão da release (o app e a CLI já carregam esses
metadados no recurso de versão do Windows).

### 3.3 Signing policy
- **Add signing policy**: slug `release-signing`, certificado da Foundation.
- **Origin verification**: ligado; repositório `LJCGJ/Dub_Aud_Vid`, branch `main`
  (as tags `v*` são criadas a partir dela).
- **Approval**: manual, você como aprovador (o programa exige aprovação humana
  por release — serão **dois pedidos** por release: executáveis e instalador).

### 3.4 Integração com o GitHub
- **Trusted build systems → GitHub Actions**: instale o GitHub App do SignPath
  no repositório (é o que fornece a verificação de origem).
- **Users → CI user**: crie um usuário de CI e gere um **API token**.

## 4. Segredos e variáveis no GitHub

Em *Settings → Secrets and variables → Actions* do repositório:

| Tipo | Nome | Valor |
|---|---|---|
| Secret | `SIGNPATH_API_TOKEN` | token do usuário de CI |
| Variable | `SIGNPATH_ORGANIZATION_ID` | GUID da organização (portal → Organization) |
| Variable | `SIGNPATH_PROJECT_SLUG` | `dub` |
| Variable | `SIGNPATH_SIGNING_POLICY_SLUG` | `release-signing` |

Assim que `SIGNPATH_ORGANIZATION_ID` existir, o job `windows-x64` do release
passa a: compilar → enviar `dub-app.exe` + `dub.exe` → **esperar sua aprovação**
no portal → montar o instalador com os executáveis assinados → enviar o
instalador → **esperar sua aprovação** → publicar. O tempo de espera por
aprovação é de até 1 h por estágio.

## 5. Testar

1. `git tag v0.1.1 && git push origin v0.1.1`
2. Acompanhe em *Actions → Release*; quando o job parar em "Assinar executáveis",
   aprove o pedido em https://app.signpath.io (Signing requests).
3. Repita para "Assinar instalador".
4. No Windows: clique com o botão direito no `Dub_0.1.1_x64-setup.exe` →
   Propriedades → aba *Assinaturas digitais* deve mostrar *SignPath Foundation*.

## Obrigações contínuas

- Manter a seção *Política de assinatura de código* no README e nas releases.
- Revisar qualquer pull request externo antes de mesclar (inclusive mudanças em
  workflows e scripts de build).
- Assinar apenas binários construídos pelo CI a partir deste repositório.
