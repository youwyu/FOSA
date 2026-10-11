// Load the 3D landscape (the three.js bundle) only when its section approaches the viewport.
(() => {
  const section = document.querySelector('#landscape');
  if (!section) return;
  const src = new URL('landscape.bundle.js', document.currentScript.src).href;
  const load = () => {
    const script = document.createElement('script');
    script.src = src;
    document.body.append(script);
  };
  if (!('IntersectionObserver' in window)) {
    load();
    return;
  }
  const observer = new IntersectionObserver(entries => {
    if (!entries.some(entry => entry.isIntersecting)) return;
    observer.disconnect();
    load();
  }, { rootMargin: '250px 0px' });
  observer.observe(section);
})();
