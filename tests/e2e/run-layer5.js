// Layer 5 runner — bypass the `playwright.cmd` → cmd.exe chain (which fails
// with `spawn C:\WINDOWS\system32\cmd.exe ENOENT` in our shell sandbox).
// Instead, spawn Node directly with the Playwright CLI's JavaScript entry
// point. Node can launch the test runner without ever touching cmd.exe.
const path = require("path");
const { spawn } = require("child_process");

const FRONTEND = "D:/prep for work/Projects/YuanRAG/src/frontend";
const TEST_CONFIG = "D:/prep for work/Projects/YuanRAG/tests/e2e/playwright.config.ts";

const nodePath = `${FRONTEND}/node_modules`;
const env = {
  ...process.env,
  NODE_PATH: nodePath,
  PLAYWRIGHT_BROWSERS_PATH: "C:/Users/ghy/AppData/Local/ms-playwright",
};

const nodeBin = "node";
const playwrightCli = path.join(
  FRONTEND,
  "node_modules",
  "@playwright",
  "test",
  "cli.js"
);

console.log(
  `Spawning: ${nodeBin} ${playwrightCli} test --config=${TEST_CONFIG} --reporter=line`
);

const child = spawn(
  nodeBin,
  [
    playwrightCli,
    "test",
    `--config=${TEST_CONFIG}`,
    "--reporter=line",
  ],
  {
    env,
    stdio: "inherit",
    shell: false,           // CRITICAL — avoid cmd.exe shim
    cwd: FRONTEND,
  }
);

child.on("exit", (code) => {
  console.log(`Playwright exited with code ${code}`);
  process.exit(code ?? 1);
});
child.on("error", (err) => {
  console.error("Spawn error:", err);
  process.exit(2);
});