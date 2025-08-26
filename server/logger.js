// logger.js - tiny leveled logger for Node and browser
const LEVELS = { error: 0, warn: 1, info: 2, debug: 3 };
const rawLevel = (typeof process !== 'undefined' && process.env && process.env.LOG_LEVEL)
  || (typeof globalThis !== 'undefined' && (globalThis.LOG_LEVEL || globalThis.logLevel))
  || 'info';
const CURRENT = LEVELS[String(rawLevel).toLowerCase()] ?? 2;

const pad = (t='') => (t && t.startsWith('[') ? t : `[${t}]`).padEnd(16);

export const log = (lvl, tag, ...args) => {
  const n = LEVELS[lvl] ?? 2;
  if (n <= CURRENT) {
    const fn = console[lvl] || console.log;
    fn(pad(tag), ...args);
  }
};

export const info  = (tag, ...args) => log('info',  tag, ...args);
export const warn  = (tag, ...args) => log('warn',  tag, ...args);
export const error = (tag, ...args) => log('error', tag, ...args);
export const debug = (tag, ...args) => log('debug', tag, ...args);
