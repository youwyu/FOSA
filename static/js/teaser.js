// Discovered-task examples: one at a time, replaying the construction as the task is shown.
(() => {
  const picker = document.querySelector('.teaser-picker');
  const examples = Array.from(document.querySelectorAll('.teaser-example'));
  if (!picker || !examples.length) return;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let active = 0, visible = true;

  const tabs = examples.map((example, index) => {
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.id = `${example.id}-tab`;
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', example.id);
    tab.innerHTML = `${example.dataset.label}<span>${example.dataset.robot} · ${example.dataset.tag}</span>`;
    example.setAttribute('role', 'tabpanel');
    example.setAttribute('aria-labelledby', tab.id);
    if (reduceMotion) example.querySelector('video').controls = true;
    tab.addEventListener('click', () => select(index));
    tab.addEventListener('keydown', event => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      select((index + step + examples.length) % examples.length, true);
    });
    picker.append(tab);
    return tab;
  });

  function sync() {
    examples.forEach((example, i) => {
      const video = example.querySelector('video');
      if (i === active && visible && !reduceMotion) video.play().catch(() => {});
      else video.pause();
    });
  }

  function select(index, moveFocus = false) {
    active = index;
    examples.forEach((example, i) => {
      example.hidden = i !== index;
      tabs[i].setAttribute('aria-selected', String(i === index));
      tabs[i].tabIndex = i === index ? 0 : -1;
    });
    const example = examples[index];
    example.classList.remove('is-animating');
    void example.offsetWidth;
    example.classList.add('is-animating');
    if (moveFocus) tabs[index].focus();
    sync();
  }

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting && entry.intersectionRatio >= .4;
      sync();
    }, { threshold: [0, .4] }).observe(document.querySelector('.teaser-examples'));
  }
  document.addEventListener('visibilitychange', () => { if (!document.hidden) sync(); });
  picker.hidden = false;
  select(0);
})();
