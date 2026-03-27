const { chromium } = require("playwright");
const fs = require("fs");
const http = require("http");

const challengeHtmlPath = process.argv[2];
const outputPath = process.argv[3] || "challenge-preview.png";

if (!challengeHtmlPath) {
  console.error("Usage: node render-challenge.js <challenge.html> [output.png]");
  process.exit(1);
}

const fragment = fs.readFileSync(challengeHtmlPath, "utf-8");

const fullHtml = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script>
    window.MathJax = {
      tex: { inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] },
      startup: {
        pageReady: () => MathJax.startup.defaultPageReady().then(() => {
          document.body.dataset.mathReady = 'true';
        })
      }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 800px;
      margin: 0 auto;
      padding: 32px;
      background: #fff;
      color: #1a1a1a;
      font-size: 15px;
      line-height: 1.6;
    }
    h2 { border-bottom: 1px solid #e0e0e0; padding-bottom: 6px; margin-top: 28px; }
    code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
    pre { background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; }
    ul { padding-left: 24px; }
    li { margin-bottom: 4px; }
    svg { max-width: 100%; height: auto; }
  </style>
</head>
<body>
${fragment}
</body>
</html>`;

(async () => {
  const server = http.createServer((req, res) => {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(fullHtml);
  });

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 900, height: 600 } });

  await page.goto(`http://127.0.0.1:${port}`, { waitUntil: "networkidle" });

  try {
    await page.waitForFunction(
      () => document.body.dataset.mathReady === "true",
      { timeout: 15000 }
    );
  } catch {
    console.warn("MathJax did not signal ready within 15s, continuing anyway");
  }

  await page.waitForTimeout(500);

  const body = page.locator("body");
  await body.screenshot({ path: outputPath, type: "png" });
  console.log("Screenshot saved to " + outputPath);

  await browser.close();
  server.close();
})();
