const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let pyProc = null;

function startPythonBackend() {
  const script = path.join(__dirname, 'backend_app.py');
  pyProc = spawn('python', [script], { cwd: __dirname, stdio: ['ignore', 'pipe', 'pipe'] });

  pyProc.stdout.on('data', (data) => {
    console.log(`PY: ${data}`);
  });
  pyProc.stderr.on('data', (data) => {
    console.error(`PY ERR: ${data}`);
  });
}

function stopPythonBackend() {
  if (pyProc) {
    try { pyProc.kill(); } catch (e) { /* ignore */ }
    pyProc = null;
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 900,
    height: 700,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  win.loadFile(path.join(__dirname, 'renderer_index.html'));
}

app.whenReady().then(() => {
  startPythonBackend();
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  stopPythonBackend();
  if (process.platform !== 'darwin') app.quit();
});

// IPC: choose directory
ipcMain.handle('choose-directory', async () => {
  const result = await dialog.showOpenDialog({ properties: ['openDirectory'] });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});
