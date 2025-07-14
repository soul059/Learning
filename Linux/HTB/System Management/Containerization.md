# Containerization in Linux

Containerization is a technology that allows packaging applications and their dependencies into isolated, lightweight environments called containers. These containers are highly portable and ensure consistent application behavior across different deployment environments by sharing the host system's kernel. Key technologies include Docker and Linux Containers (LXC).

## What is Containerization?

* **Concept:** Packaging applications with everything they need to run (code, libraries, dependencies, configurations) into a self-contained unit.
* **Isolation:** Containers run in isolated environments on a single host system, separated from the host and other containers.
* **Efficiency:** Containers share the host's kernel, making them more lightweight and resource-efficient compared to traditional Virtual Machines (VMs) which require a full operating system for each instance.
* **Benefits:** Improved security (due to isolation), portability (run consistently anywhere), scalability (easily run multiple containers), and streamlined deployment/management.
* **Analogy:** Like portable "stage pods" for bands at a concert – isolated setups that share the main stage but don't interfere with each other.

## Container Security

While isolation provides a security barrier, containers are not inherently fully secure:

* **Isolation is Lighter than VMs:** Containers share the host kernel, which is a potential attack surface if compromised.
* **Risks:** Misconfigurations can lead to vulnerabilities like privilege escalation or container escapes, potentially allowing attackers to gain access to the host system or other containers.
* **Mitigation:** Requires proper configuration, hardening, and security monitoring.

## Docker

Docker is a popular open-source platform for building, deploying, and managing applications within containers.

* **Purpose:** Automates the process of creating and deploying applications in self-contained containers using a layered filesystem and resource isolation.
* **Docker Images:** Read-only templates containing the application and all its dependencies. Images are used to create running containers.
    * Obtained from registries like **Docker Hub** (a public and private repository for images) or built custom.
* **Dockerfile:** A text file containing instructions for building a Docker image step-by-step.

    * **Example Dockerfile (for a file hosting container):**
        ```dockerfile
        # Use the latest Ubuntu 22.04 LTS as the base image
        FROM ubuntu:22.04

        # Update the package repository and install the required packages (Apache, SSH)
        RUN apt-get update && \
            apt-get install -y \
                apache2 \
                openssh-server \
                && \
            rm -rf /var/lib/apt/lists/* # Clean up apt cache

        # Create a new user and set password
        RUN useradd -m docker-user && \
            echo "docker-user:password" | chpasswd

        # Give the docker-user access/permissions and sudo rights
        RUN chown -R docker-user:docker-user /var/www/html && \
            chown -R docker-user:docker-user /var/run/apache2 && \
            chown -R docker-user:docker-user /var/log/apache2 && \
            chown -R docker-user:docker-user /var/lock/apache2 && \
            usermod -aG sudo docker-user && \
            echo "docker-user ALL=(ALL) NOPASSWD: ALL" %3E> /etc/sudoers

        # Expose the required ports (SSH and HTTP)
        EXPOSE 22 80

        # Start the SSH and Apache services when the container starts
        CMD service ssh start && /usr/sbin/apache2ctl -D FOREGROUND
        ```

* **Building a Docker Image from a Dockerfile:**
    ```bash
    docker build -t %3Ctag> <path_to_dockerfile_directory>
    ```
    Example:
    ```bash
    docker build -t FS_docker . # Builds image from Dockerfile in current directory with tag FS_docker
    ```

* **Running a Docker Container from an Image:**
    ```bash
    docker run [options] <image_name_or_id> [command]
    ```
    * `-p <host_port>:<container_port>`: Maps a port on the host to a port inside the container.
    * `-d`: Runs the container in detached (background) mode.
    * **Example (Run FS_docker, map ports, detached):**
        ```bash
        docker run -p 8022:22 -p 8080:80 -d FS_docker
        ```
        (Maps host port 8022 to container SSH port 22, and host port 8080 to container HTTP port 80)

* **Docker Management Commands:**
    * `docker ps`: List currently running containers. Use `docker ps -a` to list all containers (running and stopped).
    * `docker stop <container_id_or_name>`: Stop a running container.
    * `docker start <container_id_or_name>`: Start a stopped container.
    * `docker restart <container_id_or_name>`: Restart a container.
    * `docker rm <container_id_or_name>`: Remove a container.
    * `docker rmi <image_id_or_tag>`: Remove a Docker image.
    * `docker logs <container_id_or_name>`: View the logs of a container.

* **Data Persistence:** Containers are stateless by design. To save data, use volumes to mount directories from the host (or a volume manager) into the container.
* **Orchestration:** For managing multiple containers in production, tools like Docker Compose (for defining multi-container applications) and Kubernetes (for large-scale orchestration) are used.

## Linux Containers (LXC)

LXC is a lightweight virtualization technology that provides process and resource isolation using features built into the Linux kernel like cgroups and namespaces.

* **Purpose:** Creates isolated Linux environments that behave like lightweight VMs, sharing the host kernel.
* **Isolation:** Achieved using kernel **namespaces** (isolating processes, network, mount points) and **control groups (cgroups)** (limiting resources like CPU, memory, disk I/O).
    * **Namespaces:** Provide each container with its own isolated view of system resources (PID space, network interfaces, file system mount points).
    * **Cgroups:** Allow limiting and accounting for the resource usage of processes.

* **Installation:**
    ```bash
    sudo apt-get install lxc lxc-utils -y
    ```

* **Creating an LXC Container:**
    ```bash
    sudo lxc-create -n <container_name> -t <template>
    ```
    Example:
    ```bash
    sudo lxc-create -n linuxcontainer -t ubuntu # Creates an Ubuntu container named 'linuxcontainer'
    ```

* **Managing LXC Containers:**
    * `lxc-ls`: List existing containers.
    * `lxc-start -n <container>`: Start a container.
    * `lxc-stop -n <container>`: Stop a container.
    * `lxc-restart -n <container>`: Restart a container.
    * `lxc-config -n <container_name> -s <setting_type>`: Manage container configuration settings (storage, network, security).
    * `lxc-attach -n <container>`: Connect to a running container's console.
    * `lxc-attach -n <container> -f /path/to/share`: Connect and share a specific file/directory (advanced).

* **Configuring Resource Limits (Cgroups) for LXC:** Edit the container's configuration file (e.g., `/usr/share/lxc/config/linuxcontainer.conf`) to set resource limits.
    * **Example Configuration Snippet:**
        ```ini
        lxc.cgroup.cpu.shares = 512         # Allocate CPU shares (relative to other containers)
        lxc.cgroup.memory.limit_in_bytes = 512M # Limit memory usage
```

    * Apply changes by restarting the LXC service: `sudo systemctl restart lxc.service`.

## Docker vs. LXC

| Category       | LXC                                        | Docker                                         |
| :------------- | :----------------------------------------- | :--------------------------------------------- |
| **Approach** | System-level (like lightweight VMs)        | Application-focused (packaging single apps)    |
| **Image Build**| More manual (root filesystem, install pkgs)| Standardized (Dockerfile)                      |
| **Portability**| Less portable (tied to host config)        | Highly portable (standardized images)          |
| **Ease of Use**| Requires more Linux admin knowledge        | User-friendly CLI, large community support     |
| **Security** | Requires manual hardening/config           | More isolation out-of-the-box (AppArmor, SELinux, read-only fs) |

Both technologies are valuable, and both can potentially have privilege escalation vulnerabilities if not securely configured.

Containers are very useful in cybersecurity for creating isolated testing environments (for software, exploits, malware) without risking the host system.