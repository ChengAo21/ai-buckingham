(() => {
  const status = document.getElementById('copy-status');
  document.querySelectorAll('[data-copy]').forEach(button => {
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        button.textContent = 'Copied';
        if (status) status.textContent = 'Command copied to clipboard.';
        setTimeout(() => { button.textContent = 'Copy'; }, 1800);
      } catch {
        const code = button.parentElement.querySelector('code');
        const selection = getSelection();
        if (code && selection) {
          const range = document.createRange();
          range.selectNodeContents(code);
          selection.removeAllRanges();
          selection.addRange(range);
        }
        if (status) status.textContent = 'Command selected. Use your browser copy command.';
      }
    });
  });
})();
