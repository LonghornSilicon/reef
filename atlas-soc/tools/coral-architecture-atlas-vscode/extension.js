const vscode = require("vscode");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const model = require("./model.json");

const VIEW_ID = "coralArchitectureAtlasMentalModel.view";
const COMMAND_PREFIX = "coralArchitectureAtlasMentalModel";

function findOccurrence(text, needle, occurrence = 0) {
  let index = -1;
  let from = 0;
  for (let count = 0; count <= occurrence; count += 1) {
    index = text.indexOf(needle, from);
    if (index < 0) return -1;
    from = index + needle.length;
  }
  return index;
}

function anchorsFor(symbol) {
  const seen = new Set();
  return [symbol.open, ...(symbol.matches || [])].filter(anchor => {
    if (!anchor) return false;
    const key = `${anchor.file}\0${anchor.needle}\0${anchor.occurrence || 0}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

class AtlasProvider {
  constructor(context) {
    this.context = context;
    this.view = undefined;
    this.disposables = [];
  }

  resolveWebviewView(view) {
    this.view = view;
    const mediaRoot = vscode.Uri.joinPath(this.context.extensionUri, "media");
    view.webview.options = { enableScripts: true, localResourceRoots: [mediaRoot] };
    view.webview.html = this.getHtml(view.webview);
    this.disposables.push(view.webview.onDidReceiveMessage(async message => {
      if (message.type === "ready") this.syncFromEditor(vscode.window.activeTextEditor);
      if (message.type === "openSymbol") await this.openSymbol(message.id);
      if (message.type === "openFile") await this.openFile(message.file, message.href);
      if (message.type === "copyText" && typeof message.text === "string") await vscode.env.clipboard.writeText(message.text);
    }));
    this.syncFromEditor(vscode.window.activeTextEditor);
  }

  dispose() {
    for (const disposable of this.disposables) disposable.dispose();
  }

  async focus() {
    await vscode.commands.executeCommand(`${VIEW_ID}.focus`);
  }

  async followCursor() {
    this.syncFromEditor(vscode.window.activeTextEditor, true);
  }

  relativePath(document) {
    const folder = vscode.workspace.getWorkspaceFolder(document.uri);
    if (!folder) return "";
    return vscode.workspace.asRelativePath(document.uri, false).replaceAll("\\", "/");
  }

  syncFromEditor(editor, reveal = false) {
    if (!this.view) return;
    if (!editor) {
      this.post({ type: "editorSync", active: false, label: "No active source editor" });
      return;
    }
    const file = this.relativePath(editor.document);
    const relevant = new Set(model.symbols.flatMap(symbol => anchorsFor(symbol).map(anchor => anchor.file)));
    if (!relevant.has(file)) {
      this.post({ type: "editorSync", active: false, label: model.scopeHint });
      return;
    }
    const text = editor.document.getText();
    const cursor = editor.document.offsetAt(editor.selection.active);
    const candidates = [];
    for (const symbol of model.symbols) {
      for (const anchor of anchorsFor(symbol)) {
        if (anchor.file !== file) continue;
        const index = findOccurrence(text, anchor.needle, anchor.occurrence || 0);
        if (index >= 0 && index <= cursor) candidates.push({ symbol, index });
      }
    }
    candidates.sort((left, right) => left.index - right.index);
    const current = candidates.at(-1)?.symbol;
    this.post({ type: "editorSync", active: true, symbolId: current?.id || "", label: `${file}:${editor.selection.active.line + 1}` });
    if (reveal) this.view.show?.(true);
  }

  async resolveWorkspaceFile(relativeFile) {
    for (const folder of vscode.workspace.workspaceFolders || []) {
      const uri = vscode.Uri.joinPath(folder.uri, ...relativeFile.split("/"));
      try {
        await vscode.workspace.fs.stat(uri);
        return uri;
      } catch {
        // Continue across multi-root workspaces.
      }
    }
    return undefined;
  }

  async openSymbol(id) {
    const symbol = model.symbols.find(candidate => candidate.id === id);
    if (!symbol) return;
    const uri = await this.resolveWorkspaceFile(symbol.open.file);
    if (!uri) {
      void vscode.window.showErrorMessage(`Architecture source not found: ${symbol.open.file}`);
      return;
    }
    const document = await vscode.workspace.openTextDocument(uri);
    const index = findOccurrence(document.getText(), symbol.open.needle, symbol.open.occurrence || 0);
    if (index < 0) {
      void vscode.window.showErrorMessage(`Source anchor moved or was renamed: ${symbol.name}`);
      return;
    }
    const start = document.positionAt(index);
    const end = document.positionAt(index + symbol.open.needle.length);
    const editor = await vscode.window.showTextDocument(document, { viewColumn: vscode.ViewColumn.One, preserveFocus: false, preview: true });
    editor.selection = new vscode.Selection(start, start);
    editor.revealRange(new vscode.Range(start, end), vscode.TextEditorRevealType.InCenterIfOutsideViewport);
  }

  async openFile(relativeFile, fallbackHref) {
    if (!relativeFile) return;
    const uri = await this.resolveWorkspaceFile(relativeFile);
    if (!uri) {
      if (fallbackHref && /^https?:/i.test(fallbackHref)) await vscode.env.openExternal(vscode.Uri.parse(fallbackHref));
      else void vscode.window.showErrorMessage(`Linked document not found: ${relativeFile}`);
      return;
    }
    if (/\.(pdf|png|jpe?g)$/i.test(relativeFile)) {
      await vscode.commands.executeCommand("vscode.open", uri);
      return;
    }
    const document = await vscode.workspace.openTextDocument(uri);
    await vscode.window.showTextDocument(document, { viewColumn: vscode.ViewColumn.One, preserveFocus: false, preview: true });
  }

  post(message) {
    void this.view?.webview.postMessage(message);
  }

  getHtml(webview) {
    const nonce = crypto.randomBytes(16).toString("hex");
    const mediaRoot = vscode.Uri.joinPath(this.context.extensionUri, "media");
    const stylesUri = webview.asWebviewUri(vscode.Uri.joinPath(mediaRoot, "styles.css"));
    const appUri = webview.asWebviewUri(vscode.Uri.joinPath(mediaRoot, "app.js"));
    const template = fs.readFileSync(path.join(this.context.extensionPath, "media", "index.html"), "utf8");
    const body = template.match(/<body>([\s\S]*?)<script src="model\.js"><\/script>[\s\S]*?<script src="app\.js"><\/script>[\s\S]*?<\/body>/i)?.[1] || "<main>Viewer template missing.</main>";
    const data = JSON.stringify(model.atlas).replaceAll("<", "\\u003c");
    return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data: ${webview.cspSource}; font-src ${webview.cspSource}; style-src ${webview.cspSource}; script-src 'nonce-${nonce}' ${webview.cspSource};">
  <link rel="stylesheet" href="${stylesUri}">
  <title>Coral Architecture Atlas</title>
</head>
<body>
${body}
<script nonce="${nonce}">window.__CORAL_MODEL__ = ${data};</script>
<script nonce="${nonce}" src="${appUri}"></script>
</body>
</html>`;
  }
}

function activate(context) {
  const provider = new AtlasProvider(context);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(VIEW_ID, provider, { webviewOptions: { retainContextWhenHidden: true } }),
    vscode.commands.registerCommand(`${COMMAND_PREFIX}.focus`, () => provider.focus()),
    vscode.commands.registerCommand(`${COMMAND_PREFIX}.followCursor`, () => provider.followCursor()),
    vscode.window.onDidChangeActiveTextEditor(editor => provider.syncFromEditor(editor)),
    vscode.window.onDidChangeTextEditorSelection(event => provider.syncFromEditor(event.textEditor)),
    provider,
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
