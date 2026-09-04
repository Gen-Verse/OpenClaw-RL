const { ipcRenderer } = require('electron');

const chooseBtn = document.getElementById('choose');
const analyzeBtn = document.getElementById('analyze');
const dirSpan = document.getElementById('dir');
const status = document.getElementById('status');
const resultPre = document.getElementById('result');
const sampleDiv = document.getElementById('sample');
let selectedDir = null;

chooseBtn.addEventListener('click', async () => {
  const dir = await ipcRenderer.invoke('choose-directory');
  if (!dir) return;
  selectedDir = dir;
  dirSpan.textContent = dir;
  analyzeBtn.disabled = false;
});

analyzeBtn.addEventListener('click', async () => {
  if (!selectedDir) return;
  status.textContent = '状态：分析中…';
  resultPre.textContent = '';
  sampleDiv.textContent = '';
  try {
    const resp = await fetch('http://127.0.0.1:5000/analyze', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({dir: selectedDir})
    });
    const data = await resp.json();
    resultPre.textContent = JSON.stringify(data.analysis, null, 2);
    sampleDiv.textContent = data.sample;
    status.textContent = '状态：完成';
  } catch (e) {
    status.textContent = '状态：出错：' + e.message;
  }
});
