const { spawn } = require("child_process");
const path = require("path");

const rootDir = path.resolve(__dirname, "..");
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";

const processes = [
  spawn("node", ["backened/server.js"], {
    cwd: rootDir,
    env: { ...process.env, PORT: "3001" },
    stdio: "inherit",
    shell: true
  }),
  spawn(npmCommand, ["run", "dev:frontend"], {
    cwd: rootDir,
    stdio: "inherit",
    shell: true
  })
];

function shutdown() {
  processes.forEach((child) => {
    if (!child.killed) {
      child.kill();
    }
  });
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
