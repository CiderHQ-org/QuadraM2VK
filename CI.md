# Linux distribution CI

The `Linux distributions` workflow runs on pushes and pull requests to `quadram2vk`,
manual dispatch, and a weekly schedule. Each job builds the current workspace's Linux
host bridge and runs authored offscreen Vulkan, preload/cache/memory, launcher cancellation,
and trace-reader checks. Native Wayland test code is also compiled with warnings denied.

| Job | Container | Vulkan implementation |
| --- | --- | --- |
| Debian 13 | `debian:13-slim` | Distro lavapipe |
| Ubuntu 24.04 | `ubuntu:24.04` | Distro lavapipe |
| Ubuntu 26.04 | `ubuntu:26.04` | Distro lavapipe |
| Fedora 44 | `registry.fedoraproject.org/fedora:44` | Distro lavapipe |
| Arch rolling | `archlinux:base` | Distro lavapipe |

APT jobs use LLVM's signed LLVM 23 repository for the IR tools and legacy AIR library;
Fedora/Arch use their LLVM packages. Rust stable and its format/clippy components are installed
in each container. The Vulkan loader, Mesa implementation and validation layers remain the
distribution's own packages, so driver/package compatibility failures are visible.

Jobs have a 45-minute timeout, keep running when another distro fails, use two build jobs,
and upload the host binary plus numeric/authored test logs for 14 days. Client binaries,
shaders, caches, account data and the Darling SDK are not CI inputs. Live compositor tests,
Darling guest execution, game performance and AMD/Intel/NVIDIA hardware qualification need
separate runs; a distro CI pass describes the software driver and checks listed above.

To repeat the checks in a disposable container, mount the source workspace, set
`QM_CI_FAMILY` to `debian`, `fedora` or `arch`, install Rust stable, and run:

```sh
bash scripts/ci-distro.sh deps
bash scripts/ci-distro.sh check
```

The dependency step requires a root container. `check` does not install OS packages.
GitHub documents the [job container behavior](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/run-jobs-in-a-container);
LLVM documents its [Debian/Ubuntu package repository](https://apt.llvm.org/).
