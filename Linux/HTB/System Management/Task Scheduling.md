# Task Scheduling in Linux

Task scheduling is a powerful feature in Linux that automates the execution of commands or scripts at specified times or intervals. It is widely used for routine system maintenance, backups, updates, and custom script execution, eliminating the need for manual intervention.

## Importance for Cybersecurity

Understanding task scheduling is crucial in cybersecurity:

* **Legitimate Tool:** Used by administrators for necessary automation.
* **Malicious Vector:** Can be used by attackers to establish persistence (running malicious scripts at scheduled times) or execute harmful actions.
* **Auditing:** Examining scheduled tasks (`cron jobs`, `systemd timers`) is a key part of security audits to identify unauthorized activity.
* **Penetration Testing:** Can be leveraged to simulate attack scenarios.

Linux offers several tools for task scheduling; two prominent ones are `systemd` timers and `cron`.

## Systemd Timers

`systemd` timers are a modern method for scheduling tasks, often used in distributions that have adopted `systemd` as their init system. Scheduling with `systemd` involves creating two unit files:

1.  **Timer Unit (`.timer`)**: Defines *when* the task should run.
    * Stored typically in `/etc/systemd/system/`.
    * Contains `[Unit]`, `[Timer]`, and `[Install]` sections.
    * Key `[Timer]` options include `OnBootSec` (run after boot) and `OnUnitActiveSec` (run relative to when the associated service last became active).
    * **Example (`mytimer.timer`):**
        ```ini
        [Unit]
        Description=My Timer

        [Timer]
        OnBootSec=3min      # Run 3 minutes after boot
        OnUnitActiveSec=1hour # Run 1 hour after the service was last active

        [Install]
        WantedBy=timers.target
```

2.  **Service Unit (`.service`)**: Defines *what* command or script to execute when the timer is triggered.
    * Stored typically in `/etc/systemd/system/`.
    * Contains `[Unit]`, `[Service]`, and `[Install]` sections.
    * `[Service]` section includes `ExecStart` specifying the command to run.
    * **Example (`mytimer.service`):**

        ```ini
        [Unit]
        Description=My Service

        [Service]
        ExecStart=/full/path/to/my/script.sh # Command or script to execute

        [Install]
        WantedBy=multi-user.target # Target unit to which this service is bound
        
```

* **Managing systemd Timers:**
    * `sudo systemctl daemon-reload`: Reloads systemd configuration after creating/modifying unit files.
    * `sudo systemctl start %3Ctimer.timer%3E`: Manually starts the timer.
    * `sudo systemctl enable <timer.timer>`: Enables the timer to start automatically at boot.
    * `sudo systemctl status <timer.timer>`: Checks the status of the timer.
    * `sudo systemctl list-timers`: Lists active timers.

## Cron

Cron is a traditional and widely used utility for scheduling tasks via the `cron` daemon. Tasks are defined in a configuration file called a `crontab`.

* **Crontab Structure:** Each line in a `crontab` represents a scheduled task and follows a specific format for specifying the time and the command.
    ```
    Minutes | Hours | Day of Month | Month | Day of Week | Command to execute
    ```
	| Time Field    | Description                     | Allowed Values   |
    | :------------ | :------------------------------ | :--------------- |
	| Minutes        | The minute of the hour (0-59)   | 0-59             |
	| Hours            | The hour of the day (0-23)      | 0-23             |
	| Day of Month  | The day of the month (1-31)     | 1-31             |
	| Month         | The month of the year (1-12)    | 1-12 (or names)  |
	| Day of Week   | The day of the week (0-7)       | 0-7 (0 or 7 for Sunday, or names) |

* **Examples of Crontab Entries:**
    * `0 0 1 * * /path/to/scripts/run_scripts.sh`: Runs `/path/to/scripts/run_scripts.sh` at midnight (00:00) on the 1st day of every month.
    * `0 0 * * 0 /path/to/scripts/clean_database.sh`: Runs `/path/to/scripts/clean_database.sh` at midnight (00:00) every Sunday (`0` or `7`).
    * `0 0 * * 7 /path/to/scripts/backup.sh`: Also runs `/path/to/scripts/backup.sh` at midnight every Sunday (`7`).

* **Managing Crontab:** Use the `crontab` command to edit, list, or remove user crontab files.
    * `crontab -e`: Edit the current user's crontab.
    * `crontab -l`: List the current user's crontab entries.
    * `crontab -r`: Remove the current user's crontab.

## Systemd Timers vs. Cron

Both tools automate tasks, but differ in configuration and capabilities:

* **Configuration:** systemd timers use separate `.timer` and `.service` unit files, offering more flexibility and integration with systemd's features. Cron uses entries within a single `crontab` file.
* **Capabilities:** systemd timers can be scheduled based on more complex events and states than Cron's time-based scheduling.

Understanding both systemd timers and Cron is important for comprehensive task automation and for identifying potentially malicious scheduled activities on a Linux system.