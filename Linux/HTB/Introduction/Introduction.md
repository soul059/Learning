# Understanding Linux: A Pillar in Cybersecurity

Linux is a powerful and flexible operating system widely used across personal computers, servers, and mobile devices. Its significance in cybersecurity stems from its robustness, open-source nature, and adaptability. This summary covers the foundational aspects discussed: its structure, history, underlying philosophy, architecture, and file system hierarchy.

## What is Linux?

At its core, Linux is an **operating system (OS)**, similar to Windows or macOS. An OS is the essential software that manages a computer's hardware resources, enabling communication between software applications and the hardware components.

A key characteristic of Linux is the existence of numerous **distributions (distros)**. These are different versions of Linux, bundled with specific software and configurations, tailored for various purposes and user preferences. Popular examples include Ubuntu, Debian, Fedora, RedHat, and many more.

## A Brief History

The lineage of Linux traces back to the **Unix operating system**, released in 1970. Early developments like the Berkeley Software Distribution (BSD) and Richard Stallman's **GNU project** (started in 1983 with the goal of a free Unix-like OS, leading to the **GNU General Public License - GPL**) paved the way.

The **Linux kernel** itself was a personal project initiated in 1991 by **Linus Torvalds**. Initially small and non-commercially licensed, it has grown immensely and is now licensed under the **GNU GPL v2**, allowing for free modification and distribution. Today, there are over 600 Linux distributions available.

## Philosophy, Architecture, and File System Hierarchy

While not explicitly detailed as separate technical sections in the text, the philosophy and architecture can be understood through an analogy:

* **Philosophy:** Think of Linux's philosophy as a company's culture, guided by principles like **simplicity, transparency, and cooperation**. Its open-source nature embodies this, allowing anyone to view, modify, and distribute the code.
* **Architecture:** This represents the organizational structure of Linux's components – like departments in a company – outlining how they are arranged and communicate for efficient operation. (Note: The provided text does not delve into the technical specifics of the architecture).
* **File System Hierarchy:** This refers to the way files and directories are organized in Linux, typically in a tree-like structure starting from the root (`/`). This is a fundamental concept, though not elaborated upon in the source text.

## Security, Stability, and Usage

Linux is generally considered **more secure** than many other operating systems, being less susceptible to malware, frequently updated, and offering high stability and performance. However, it can present a steeper learning curve for beginners and may have fewer hardware drivers compared to Windows.

Due to its **free and open-source** nature, Linux can be modified and distributed freely. This has led to its widespread adoption across:

* Servers and Mainframes
* Desktops
* Embedded Systems (routers, TVs, game consoles)
* Mobile Devices (Android is based on the Linux kernel, making Linux the most widely installed OS globally)

## Parrot OS

The interactive instances mentioned provide access to **Pwnbox**, which runs a customized version of **Parrot OS**. This is a Debian-based Linux distribution specifically designed with a focus on **security, privacy, and development**, making it a suitable environment for cybersecurity professionals.

Understanding these foundational elements of Linux provides a crucial starting point for navigating its environment and leveraging its capabilities, particularly within the field of cybersecurity.

## Philosophy

The Linux philosophy centers on simplicity, modularity, and openness. It advocates for building small, single-purpose programs that perform one task well. These programs can be combined in various ways to accomplish complex operations, promoting efficiency and flexibility. Linux follows five core principles:

|**Principle**|**Description**|
|---|---|
|`Everything is a file`|All configuration files for the various services running on the Linux operating system are stored in one or more text files.|
|`Small, single-purpose programs`|Linux offers many different tools that we will work with, which can be combined to work together.|
|`Ability to chain programs together to perform complex tasks`|The integration and combination of different tools enable us to carry out many large and complex tasks, such as processing or filtering specific data results.|
|`Avoid captive user interfaces`|Linux is designed to work mainly with the shell (or terminal), which gives the user greater control over the operating system.|
|`Configuration data stored in a text file`|An example of such a file is the `/etc/passwd` file, which stores all users registered on the system.|

---

## Components

|**Component**|**Description**|
|---|---|
|`Bootloader`|A piece of code that runs to guide the booting process to start the operating system. Parrot Linux uses the GRUB Bootloader.|
|`OS Kernel`|The kernel is the main component of an operating system. It manages the resources for system's I/O devices at the hardware level.|
|`Daemons`|Background services are called "daemons" in Linux. Their purpose is to ensure that key functions such as scheduling, printing, and multimedia are working correctly. These small programs load after we booted or log into the computer.|
|`OS Shell`|The operating system shell or the command language interpreter (also known as the command line) is the interface between the OS and the user. This interface allows the user to tell the OS what to do. The most commonly used shells are Bash, Tcsh/Csh, Ksh, Zsh, and Fish.|
|`Graphics server`|This provides a graphical sub-system (server) called "X" or "X-server" that allows graphical programs to run locally or remotely on the X-windowing system.|
|`Window Manager`|Also known as a graphical user interface (GUI). There are many options, including GNOME, KDE, MATE, Unity, and Cinnamon. A desktop environment usually has several applications, including file and web browsers. These allow the user to access and manage the essential and frequently accessed features and services of an operating system.|
|`Utilities`|Applications or utilities are programs that perform particular functions for the user or another program.|

---

## Linux Architecture

The Linux operating system can be broken down into layers:

|**Layer**|**Description**|
|---|---|
|`Hardware`|Peripheral devices such as the system's RAM, hard drive, CPU, and others.|
|`Kernel`|The core of the Linux operating system whose function is to virtualize and control common computer hardware resources like CPU, allocated memory, accessed data, and others. The kernel gives each process its own virtual resources and prevents/mitigates conflicts between different processes.|
|`Shell`|A command-line interface (**CLI**), also known as a shell that a user can enter commands into to execute the kernel's functions.|
|`System Utility`|Makes available to the user all of the operating system's functionality.|

---

## File System Hierarchy

The Linux operating system is structured in a tree-like hierarchy and is documented in the [Filesystem Hierarchy](http://www.pathname.com/fhs/) Standard (FHS). Linux is structured with the following standard top-level directories:

![Diagram of Linux file system hierarchy with root directory branching to folders: /bin, /boot, /dev, /etc, /lib, /media, /mnt, /opt, /home, /run, /root, /proc, /sys, /tmp, /usr, /var.](https://academy.hackthebox.com/storage/modules/18/NEW_filesystem.png)

| **Path** | **Description**                                                                                                                                                                                                                                                                                                                    |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/`      | The top-level directory is the root filesystem and contains all of the files required to boot the operating system before other filesystems are mounted, as well as the files required to boot the other filesystems. After boot, all of the other filesystems are mounted at standard mount points as subdirectories of the root. |
| `/bin`   | Contains essential command binaries.                                                                                                                                                                                                                                                                                               |
| `/boot`  | Consists of the static bootloader, kernel executable, and files required to boot the Linux OS.                                                                                                                                                                                                                                     |
| `/dev`   | Contains device files to facilitate access to every hardware device attached to the system.                                                                                                                                                                                                                                        |
| `/etc`   | Local system configuration files. Configuration files for installed applications may be saved here as well.                                                                                                                                                                                                                        |
| `/home`  | Each user on the system has a subdirectory here for storage.                                                                                                                                                                                                                                                                       |
| `/lib`   | Shared library files that are required for system boot.                                                                                                                                                                                                                                                                            |
| `/media` | External removable media devices such as USB drives are mounted here.                                                                                                                                                                                                                                                              |
| `/mnt`   | Temporary mount point for regular filesystems.                                                                                                                                                                                                                                                                                     |
| `/opt`   | Optional files such as third-party tools can be saved here.                                                                                                                                                                                                                                                                        |
| `/root`  | The home directory for the root user.                                                                                                                                                                                                                                                                                              |
| `/sbin`  | This directory contains executables used for system administration (binary system files).                                                                                                                                                                                                                                          |
| `/tmp`   | The operating system and many programs use this directory to store temporary files. This directory is generally cleared upon system boot and may be deleted at other times without any warning.                                                                                                                                    |
| `/usr`   | Contains executables, libraries, man files, etc.                                                                                                                                                                                                                                                                                   |
| `/var`   | This directory contains variable data files such as log files, email in-boxes, web application related files, cron files, and more.                                                                                                                                                                                                |