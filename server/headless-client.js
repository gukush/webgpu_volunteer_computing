#!/usr/bin/env node
import puppeteer from 'puppeteer';
import minimist from 'minimist';

// This is the function that will be imported by server.js
export async function spawnPuppeteerWorkers(targetUrl, numInstances, runHeadful = false) {
  const browser = await puppeteer.launch({
    headless: runHeadful ? false : 'new',
    // WebGPU needs a *real* GPU context, so we keep GPU on and pass the
    // Chrome flags that enable WebGPU in headless mode:
    args: [
      '--enable-features=Vulkan,UseSkiaRenderer,CanvasOOPRasterization,SharedArrayBuffer',
      '--enable-unsafe-webgpu',               // Chrome ≥122 (still needed for headless)
      '--no-sandbox',                         // easier for containers
      // '--disable-dev-shm-usage',           // Recommended for Docker if /dev/shm is small
      // '--use-gl=angle',                   // Or egl or desktop, depending on the environment
      // '--use-angle=vulkan',               // If ANGLE should use Vulkan (Linux)
    ],
  });

  console.info(`Puppeteer master process launched. Spawning ${numInstances} worker page(s)...`);

  // Spin up N isolated pages
  for (let i = 0; i < numInstances; i++) {
    try {
      const page = await browser.newPage();
      // Pass a query-param so the existing client code can auto-join & stay quiet
      await page.goto(`${targetUrl}?mode=headless&workerId=${i + 1}`);
      console.info(`Puppeteer worker #${i + 1} connected → ${targetUrl}?mode=headless&workerId=${i+1}`);

      page.on('console', msg => {
        const type = msg.type();
        const text = msg.text();
        if (text.startsWith('Puppeteer worker #')) return; // Avoid self-logging
        console.log(`[Worker ${i+1} ${type}]: ${text}`);
      });
      page.on('error', err => console.error(`[Worker ${i+1} Error]: ${err.message}`, err));
      page.on('pageerror', pageErr => console.error(`[Worker ${i+1} PageError]: ${pageErr.message}`, pageErr));

      // Optional: auto-reconnect logic can be added here
      // page.on('close', async () => {
      //   console.info(`Puppeteer worker #${i + 1} closed. Attempting to respawn...`);
      //   // Add logic to respawn or handle gracefully
      // });

    } catch (e) {
      console.error(`Error spawning Puppeteer worker #${i + 1}: ${e.message}`);
    }
  }

  // Keep the browser running, or close it after some condition
  // For continuous volunteer computing, you'd keep it open.
  // browser.close(); // Example: close after setup if not continuous
}


// This part allows running headless-client.js directly from CLI
// It checks if the script is the main module being run.
async function main() {
    const argv = minimist(process.argv.slice(2));
    const url = argv._[0] ?? 'http://localhost:3000/';
    const count = Number(argv.instances ?? 1); // --instances 4
    const headful = argv.headful ? true : false;  // opt-in with --headful

    if (argv.help || argv.h) {
        console.log("Usage: node headless-client.js [url] [--instances N] [--headful]");
        console.log("\nOptions:");
        console.log("  url                URL of the WebGPU volunteer computing page (default: http://localhost:3000/)");
        console.log("  --instances N      Number of headless browser instances to launch (default: 1)");
        console.log("  --headful          Run Chrome in headful mode instead of headless (for debugging)");
        process.exit(0);
    }

    await spawnPuppeteerWorkers(url, count, headful);
}

// Check if the script is executed directly
import { fileURLToPath } from 'url';
if (process.argv[1] === fileURLToPath(import.meta.url)) {
    main().catch(error => {
        console.error("Error in headless client:", error);
        process.exit(1);
    });
}