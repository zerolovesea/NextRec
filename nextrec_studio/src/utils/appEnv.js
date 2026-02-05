const rawEnv =
  (typeof __APP_ENV__ !== 'undefined' && __APP_ENV__) ||
  (import.meta && import.meta.env && (import.meta.env.ENV || import.meta.env.VITE_ENV)) ||
  '';

export const appEnv = String(rawEnv || '').trim();
export const isIdtank = appEnv.toUpperCase() === 'IDTANK';
