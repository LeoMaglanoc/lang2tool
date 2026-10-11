// Locate a recording's bracketing frames using actual simulation timestamps.
export function frameAt(times, time) {
  if (!times.length) throw new Error('Empty recording');
  let low = 0, high = times.length - 1;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (times[mid] <= time) low = mid; else high = mid - 1;
  }
  const next = Math.min(low + 1, times.length - 1);
  const span = times[next] - times[low];
  return {index: low, next, alpha: span > 0 ? Math.max(0, Math.min(1, (time - times[low]) / span)) : 0};
}
// Do not interpolate resets, jumps, or changes in terminal/reset status.
export function continuous(a, b, eventA, eventB) {
  return Math.hypot(b[0]-a[0], b[1]-a[1], b[2]-a[2]) < .12 &&
    !(eventA !== eventB && /reset|drop|stop|failure/i.test(`${eventA} ${eventB}`));
}
// Keep time bounded and stop at the last recorded sample without silently looping.
export function advance(time, delta, speed, duration) {
  return Math.max(0, Math.min(duration, time + Math.max(0, delta) * speed));
}
