# Service and Process Management in Linux

Linux systems rely heavily on services (daemons) and processes to perform tasks in the background and run applications. Understanding how to manage these is crucial for system administration, troubleshooting, and security.

## Services (Daemons) and Processes

* **Services (Daemons):** Programs that run silently in the background, without direct user interaction, to perform system functions or provide features. They often have names ending in `d` (e.g., `sshd`, `systemd`).
    * **System Services:** Essential services needed during system boot for core functionality.
    * **User-Installed Services:** Services added by users for specific applications or features.
* **Processes:** Instances of a running program. Each process is assigned a unique **Process ID (PID)** and may have a **Parent Process ID (PPID)** if started by another process. Information about processes can be found in the `/proc/` directory.

## systemd: The Modern Init System

Most modern Linux distributions use `systemd` as the initialization system (init system). It's the first process to start during boot (PID 1) and is responsible for starting and managing other services and processes.

* **Managing Services with `systemctl`:** The primary command for interacting with systemd services.
    * `systemctl start %3Cservice%3E`: Starts a service.
        ```bash
        systemctl start ssh
        ```
    * `systemctl status <service>`: Checks the current status of a service (active, inactive, running PID, logs).
        ```bash
        systemctl status ssh
        ```
    * `systemctl enable <service>`: Configures a service to start automatically on boot.
        ```bash
        systemctl enable ssh
        ```
    * `systemctl disable <service>`: Configures a service *not* to start automatically on boot.
    * `systemctl restart <service>`: Restarts a running service.
    * `systemctl stop <service>`: Stops a running service.
    * `systemctl list-units --type=service`: Lists all loaded service units and their status.

## Viewing Logs (`journalctl`)

Logs are essential for troubleshooting service issues.

* **`journalctl -u <service>`:** Displays log entries for a specific systemd service.
    ```bash
    journalctl -u ssh.service --no-pager # --no-pager to display all output directly
    ```

## Process Control

Commands to manage the state of running processes.

* **Process States:** Processes can be Running, Waiting, Stopped, or Zombie.
* **Signals:** Commands like `kill` send signals to processes to control them.
    * `kill -l`: Lists all available signals.
    * **Common Signals:**
        * `1) SIGHUP`: Hang up.
        * `2) SIGINT`: Interrupt (`[Ctrl] + C`).
        * `9) SIGKILL`: Force kill (cannot be caught or ignored).
        * `15) SIGTERM`: Terminate (graceful shutdown request, can be handled by the process).
        * `19) SIGSTOP`: Stop the process (cannot be handled).
        * `20) SIGTSTP`: Suspend process (`[Ctrl] + Z`, can be handled).
* **`kill <signal_number> <PID>`:** Sends a specific signal to a process identified by its PID.
    ```bash
    kill 9 <PID> # Forcefully kill process with PID
    ```
* **Other Process Control Tools:** `pkill`, `pgrep`, `killall` are also used for finding and signaling processes, often by name or other attributes.
    * `pgrep <name>`: Finds PIDs of processes by name.
    * `pkill <name>`: Kills processes by name.
    * `killall <name>`: Kills processes by name.

## Managing Background and Foreground Processes

Control processes running in the current terminal session.

* **Suspending:**
    * `[Ctrl] + Z`: Suspends the currently running foreground process.
* **Listing Jobs (`jobs`):** Displays processes running or suspended in the background of the current shell session.
    ```bash
    jobs
    ```
* **Backgrounding:**
    * `bg <job_ID>`: Resumes a suspended job in the background.
    * `<command> &`: Starts a command directly in the background.
        ```bash
        ping -c 10 www.hackthebox.eu &
        ```
* **Foregrounding:**
    * `fg <job_ID>`: Brings a background or suspended job to the foreground.
        ```bash
        fg 1 # Bring job number 1 to the foreground
        ```

## Executing Multiple Commands

Combine commands on a single line.

* **`;` (Semicolon):** Separates commands. Executes commands sequentially, regardless of success or failure.
    ```bash
    echo '1'; ls MISSING_FILE; echo '3'
    ```
* **`&&` (Double Ampersand):** Executes the next command *only if* the previous command succeeded (exits with status 0).
    ```bash
    echo '1' && ls EXISTING_FILE && echo '3'
    ```
    ```bash
    echo '1' && ls MISSING_FILE && echo '3' # The third echo will not run
    ```
* **`|` (Pipe):** Connects the STDOUT of the first command to the STDIN of the second command (covered in Filtering, also allows executing multiple commands).

Understanding how to manage services and processes, including controlling their execution and state, is fundamental for operating and securing Linux systems.>)