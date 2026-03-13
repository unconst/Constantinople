// PM2 Ecosystem — Inference Subnet Services
// Usage: pm2 start ecosystem.config.js
// Status: pm2 list
// Logs: pm2 logs

module.exports = {
  apps: [
    {
      name: "gateway",
      script: "hardened_gateway.py",
      interpreter: "python3",
      cwd: __dirname,
      args: "--miners http://127.0.0.1:9502 http://176.131.50.28:40000 --port 8081 --epoch-length 120 --synthetic-interval 30 --wallet validator --hotkey default --netuid 97 --network finney --discover",
      restart_delay: 3000,
      max_restarts: 50,
      min_uptime: 10000,
      autorestart: true,
      exp_backoff_restart_delay: 1000,
      error_file: "/tmp/gateway-err.log",
      out_file: "/tmp/gateway-out.log",
      merge_logs: true,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    {
      name: "miner",
      script: "mock_miner_inline.py",
      interpreter: "python3",
      cwd: __dirname,
      args: "--port 9502",
      restart_delay: 2000,
      max_restarts: 50,
      min_uptime: 5000,
      autorestart: true,
      exp_backoff_restart_delay: 1000,
      error_file: "/tmp/miner-err.log",
      out_file: "/tmp/miner-out.log",
      merge_logs: true,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    {
      name: "monitor",
      script: "monitor.py",
      interpreter: "python3",
      cwd: __dirname,
      args: "--gateway http://127.0.0.1:8081 --interval 30 --arbos-dir /Arbos",
      restart_delay: 5000,
      max_restarts: 50,
      min_uptime: 10000,
      autorestart: true,
      exp_backoff_restart_delay: 1000,
      error_file: "/tmp/monitor-err.log",
      out_file: "/tmp/monitor-out.log",
      merge_logs: true,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    {
      name: "watchdog",
      script: "watchdog.py",
      interpreter: "python3",
      cwd: __dirname,
      args: "--interval 60 --arbos-dir /Arbos",
      restart_delay: 5000,
      max_restarts: 20,
      min_uptime: 10000,
      autorestart: true,
      error_file: "/tmp/watchdog-err.log",
      out_file: "/tmp/watchdog-out.log",
      merge_logs: true,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
  ],
};
