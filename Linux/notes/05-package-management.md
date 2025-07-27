# Package Management

## Overview

Package management is the process of installing, updating, configuring, and removing software packages on a Linux system. Different Linux distributions use different package management systems.

## Package Management Systems

### Red Hat-based Systems (RPM)
- **Distributions**: RHEL, CentOS, Fedora, openSUSE
- **Package Format**: .rpm (Red Hat Package Manager)
- **Tools**: rpm, yum, dnf, zypper

### Debian-based Systems (DEB)
- **Distributions**: Debian, Ubuntu, Linux Mint
- **Package Format**: .deb (Debian Package)
- **Tools**: dpkg, apt, apt-get, aptitude

### Arch Linux
- **Package Format**: .pkg.tar.xz
- **Tools**: pacman, makepkg

### Other Systems
- **Gentoo**: Portage (emerge)
- **Alpine**: apk
- **Snap**: Universal packages
- **Flatpak**: Universal packages
- **AppImage**: Portable applications

## APT (Advanced Package Tool) - Debian/Ubuntu

### Basic APT Commands

#### Package Installation
```bash
# Update package database
sudo apt update

# Install package
sudo apt install package_name

# Install multiple packages
sudo apt install package1 package2 package3

# Install specific version
sudo apt install package_name=version

# Install local .deb file
sudo apt install ./package.deb

# Reinstall package
sudo apt install --reinstall package_name
```

#### Package Removal
```bash
# Remove package (keep configuration files)
sudo apt remove package_name

# Remove package and configuration files
sudo apt purge package_name

# Remove unused dependencies
sudo apt autoremove

# Remove unused dependencies and configuration files
sudo apt autoremove --purge
```

#### Package Updates
```bash
# Update package database
sudo apt update

# Upgrade all packages
sudo apt upgrade

# Upgrade with new dependencies
sudo apt full-upgrade

# Upgrade specific package
sudo apt install --only-upgrade package_name
```

#### Package Information
```bash
# Search for packages
apt search keyword

# Show package information
apt show package_name

# List installed packages
apt list --installed

# List upgradable packages
apt list --upgradable

# Check package dependencies
apt depends package_name

# Check reverse dependencies
apt rdepends package_name
```

### APT Configuration

#### Sources List
```bash
# Main sources file
/etc/apt/sources.list

# Additional sources
/etc/apt/sources.list.d/

# Example sources.list entry
deb http://archive.ubuntu.com/ubuntu/ focal main restricted
deb-src http://archive.ubuntu.com/ubuntu/ focal main restricted
```

#### Repository Types
- **main**: Officially supported software
- **restricted**: Proprietary drivers
- **universe**: Community-maintained software
- **multiverse**: Software with copyright/legal restrictions

### APT Advanced Features

#### Holding Packages
```bash
# Hold package from updates
sudo apt-mark hold package_name

# Unhold package
sudo apt-mark unhold package_name

# Show held packages
apt-mark showhold
```

#### Package Pinning
```bash
# Create preferences file
sudo nano /etc/apt/preferences.d/package_name

# Example pinning
Package: package_name
Pin: version 1.2.3
Pin-Priority: 1001
```

## YUM/DNF - Red Hat/Fedora

### DNF (Dandified YUM) - Modern

#### Basic DNF Commands
```bash
# Install package
sudo dnf install package_name

# Remove package
sudo dnf remove package_name

# Update package database
sudo dnf check-update

# Update all packages
sudo dnf update

# Update specific package
sudo dnf update package_name

# Search packages
dnf search keyword

# Show package info
dnf info package_name

# List installed packages
dnf list installed

# List available packages
dnf list available
```

#### DNF Groups
```bash
# List package groups
dnf group list

# Install package group
sudo dnf group install "Group Name"

# Remove package group
sudo dnf group remove "Group Name"

# Show group info
dnf group info "Group Name"
```

### YUM (Legacy)

#### Basic YUM Commands
```bash
# Install package
sudo yum install package_name

# Remove package
sudo yum remove package_name

# Update all packages
sudo yum update

# Search packages
yum search keyword

# Show package info
yum info package_name

# List installed packages
yum list installed

# Clean cache
sudo yum clean all
```

### Repository Management

#### Adding Repositories
```bash
# Add repository (DNF)
sudo dnf config-manager --add-repo URL

# Add EPEL repository
sudo dnf install epel-release

# Repository configuration files
/etc/yum.repos.d/
```

#### Repository Configuration
```ini
[repository_name]
name=Repository Description
baseurl=http://repository.url/
enabled=1
gpgcheck=1
gpgkey=http://repository.url/RPM-GPG-KEY
```

## Pacman - Arch Linux

### Basic Pacman Commands
```bash
# Update package database
sudo pacman -Sy

# Install package
sudo pacman -S package_name

# Remove package
sudo pacman -R package_name

# Remove package and dependencies
sudo pacman -Rs package_name

# Update system
sudo pacman -Syu

# Search packages
pacman -Ss keyword

# Show package info
pacman -Si package_name

# List installed packages
pacman -Q

# List orphaned packages
pacman -Qdt
```

### AUR (Arch User Repository)
```bash
# Install AUR helper (yay)
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si

# Install AUR package
yay -S package_name

# Update AUR packages
yay -Syu
```

## Snap Packages

### Snap Commands
```bash
# Install snap
sudo apt install snapd  # On Ubuntu/Debian

# Install snap package
sudo snap install package_name

# List installed snaps
snap list

# Update snaps
sudo snap refresh

# Remove snap
sudo snap remove package_name

# Search snaps
snap find keyword

# Show snap info
snap info package_name
```

### Snap Channels
```bash
# Install from specific channel
sudo snap install package_name --channel=edge

# Available channels: stable, candidate, beta, edge
```

## Flatpak

### Flatpak Commands
```bash
# Install flatpak
sudo apt install flatpak  # On Ubuntu/Debian

# Add Flathub repository
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo

# Install application
flatpak install flathub app.id

# Run application
flatpak run app.id

# List installed apps
flatpak list

# Update apps
flatpak update

# Remove app
flatpak uninstall app.id
```

## Low-Level Package Tools

### dpkg (Debian Package)
```bash
# Install .deb package
sudo dpkg -i package.deb

# Remove package
sudo dpkg -r package_name

# List installed packages
dpkg -l

# Show package info
dpkg -s package_name

# List package files
dpkg -L package_name

# Find package owning file
dpkg -S /path/to/file

# Fix broken dependencies
sudo apt-get install -f
```

### rpm (Red Hat Package Manager)
```bash
# Install .rpm package
sudo rpm -i package.rpm

# Upgrade package
sudo rpm -U package.rpm

# Remove package
sudo rpm -e package_name

# Query installed packages
rpm -qa

# Show package info
rpm -qi package_name

# List package files
rpm -ql package_name

# Find package owning file
rpm -qf /path/to/file
```

## Package Management Best Practices

### Security
1. **Verify signatures**: Always verify package signatures
2. **Trusted repositories**: Only use official repositories
3. **Regular updates**: Keep system updated
4. **Minimal installation**: Install only necessary packages

### System Maintenance
```bash
# Clean package cache (APT)
sudo apt clean
sudo apt autoclean

# Clean package cache (DNF)
sudo dnf clean all

# Remove orphaned packages (APT)
sudo apt autoremove

# Remove orphaned packages (Pacman)
sudo pacman -Rs $(pacman -Qtdq)
```

### Dependency Management
1. **Understand dependencies**: Know what packages depend on others
2. **Avoid breaking dependencies**: Be careful when removing packages
3. **Use package managers**: Don't install software manually when packages exist
4. **Check conflicts**: Look for package conflicts before installation

## Troubleshooting Package Issues

### Common Problems

#### Broken Dependencies
```bash
# Fix broken dependencies (APT)
sudo apt install -f
sudo apt --fix-broken install

# Fix broken dependencies (DNF)
sudo dnf check
sudo dnf repoquery --unsatisfied
```

#### Locked Database
```bash
# Remove lock files (APT)
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock*

# Reconfigure dpkg
sudo dpkg --configure -a
```

#### Repository Issues
```bash
# Update repository keys (APT)
sudo apt update --allow-unauthenticated
sudo apt-key update

# Clear repository cache
sudo apt clean
sudo apt update
```

### Package Verification
```bash
# Verify package integrity (dpkg)
sudo dpkg --verify package_name

# Verify package files (rpm)
rpm -V package_name

# Check package signature (rpm)
rpm --checksig package.rpm
```

## Alternative Installation Methods

### Compiling from Source
```bash
# Typical source installation
./configure
make
sudo make install

# With custom prefix
./configure --prefix=/usr/local
make
sudo make install
```

### Python Packages (pip)
```bash
# Install Python package
pip install package_name

# Install for specific user
pip install --user package_name

# Install from requirements file
pip install -r requirements.txt

# Uninstall package
pip uninstall package_name
```

### Node.js Packages (npm)
```bash
# Install Node.js package globally
sudo npm install -g package_name

# Install locally
npm install package_name

# Uninstall package
npm uninstall package_name
```

## Package Management Tools Comparison

| Feature | APT | DNF/YUM | Pacman | Snap | Flatpak |
|---------|-----|---------|---------|------|---------|
| Install | apt install | dnf install | pacman -S | snap install | flatpak install |
| Remove | apt remove | dnf remove | pacman -R | snap remove | flatpak uninstall |
| Update | apt update && apt upgrade | dnf update | pacman -Syu | snap refresh | flatpak update |
| Search | apt search | dnf search | pacman -Ss | snap find | flatpak search |
| Info | apt show | dnf info | pacman -Si | snap info | flatpak info |
| List | apt list --installed | dnf list installed | pacman -Q | snap list | flatpak list |

## Creating Packages

### Creating .deb Packages
```bash
# Create package structure
mkdir -p package_name/DEBIAN
mkdir -p package_name/usr/bin

# Create control file
cat > package_name/DEBIAN/control << EOF
Package: package_name
Version: 1.0
Section: utils
Priority: optional
Architecture: amd64
Maintainer: Your Name <email@example.com>
Description: Package description
EOF

# Build package
dpkg-deb --build package_name
```

### Creating .rpm Packages
```bash
# Install development tools
sudo dnf install rpm-build rpmdevtools

# Create RPM build environment
rpmdev-setuptree

# Create spec file
rpmdev-newspec package_name

# Build package
rpmbuild -ba package_name.spec
```
