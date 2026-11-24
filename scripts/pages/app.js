(() => {
  const tabs = [...document.querySelectorAll('.case-option')];
  const panels = [...document.querySelectorAll('.case-panel')];
  const byId = new Map(panels.map(panel => [panel.id, panel]));

  function selectCase(id, { focus = false, updateHash = false } = {}) {
    if (!byId.has(id)) return;
    panels.forEach(panel => { panel.hidden = panel.id !== id; });
    tabs.forEach(tab => {
      const selected = tab.getAttribute('aria-controls') === id;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected && focus) tab.focus();
    });
    if (updateHash) history.replaceState(null, '', `#${id}`);
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', event => {
      event.preventDefault();
      selectCase(tab.getAttribute('aria-controls'), { updateHash: true });
    });
    tab.addEventListener('keydown', event => {
      let target;
      if (event.key === 'ArrowDown' || event.key === 'ArrowRight') target = (index + 1) % tabs.length;
      else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') target = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === 'Home') target = 0;
      else if (event.key === 'End') target = tabs.length - 1;
      else return;
      event.preventDefault();
      selectCase(tabs[target].getAttribute('aria-controls'), { focus: true, updateHash: true });
    });
  });

  function fromHash() {
    const id = location.hash.slice(1);
    if (byId.has(id)) selectCase(id);
  }
  const first = byId.has(location.hash.slice(1)) ? location.hash.slice(1) : 'case-mhd';
  selectCase(first);
  addEventListener('hashchange', fromHash);
  document.querySelectorAll('a[href^="#case-"]').forEach(link => {
    if (!link.classList.contains('case-option')) link.addEventListener('click', () => {
      selectCase(link.hash.slice(1));
    });
  });

  const status = document.getElementById('copy-status');
  document.querySelectorAll('[data-copy]').forEach(button => {
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        button.textContent = 'Copied';
        status.textContent = 'Command copied to clipboard.';
        setTimeout(() => { button.textContent = 'Copy'; }, 1800);
      } catch {
        const code = button.parentElement.querySelector('code');
        const selection = getSelection();
        const range = document.createRange();
        range.selectNodeContents(code);
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = 'Command selected. Use your browser copy command.';
      }
    });
  });
})();
