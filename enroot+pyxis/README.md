# Enroot and pyxis testing

```bash
sudo srun --container-image=ubuntu grep PRETTY /etc/os-release
```

- ERROR cannot create directory ‘/run/user/1000’: Permission denied
    
    ```bash
    $ srun --container-image=ubuntu grep PRETTY /etc/os-release
    PRETTY_NAME="CentOS Linux 8 (Core)"
    pyxis: importing docker image: centos
    slurmstepd: error: pyxis: child 15352 failed with error code: 1
    slurmstepd: error: pyxis: failed to import docker image
    slurmstepd: error: pyxis: printing enroot log file:
    slurmstepd: error: pyxis:     mkdir: cannot create directory ‘/run/user/1000’: Permission denied
    slurmstepd: error: pyxis: couldn't start container
    slurmstepd: error: spank: required plugin spank_pyxis.so: task_init() failed with rc=-1
    slurmstepd: error: Failed to invoke spank plugin stack
    slurmstepd: error: pyxis: child 15359 failed with error code: 1
    srun: error: pyxis-testing-1: task 0: Exited with exit code 1
    ```
    
    Solution: mkdir permissions → change `/etc/enroot/enroot.conf` variables to be in a user writable space:
    
    default config:
    
    ```bash
    #ENROOT_LIBRARY_PATH        /usr/lib/enroot
    #ENROOT_SYSCONF_PATH        /etc/enroot
    #ENROOT_RUNTIME_PATH        ${XDG_RUNTIME_DIR}/enroot
    #ENROOT_CONFIG_PATH         ${XDG_CONFIG_HOME}/enroot
    #ENROOT_CACHE_PATH          ${XDG_CACHE_HOME}/enroot
    #ENROOT_DATA_PATH           ${XDG_DATA_HOME}/enroot
    #ENROOT_TEMP_PATH           ${TMPDIR:-/tmp}
    ```
    
    to:
    
    ```bash
    ENROOT_LIBRARY_PATH        /home/ubuntu/enroot/library
    ENROOT_SYSCONF_PATH        /home/ubuntu/enroot/sysconf
    ENROOT_RUNTIME_PATH        /home/ubuntu/enroot/runtime
    ENROOT_CONFIG_PATH         /home/ubuntu/enroot/config
    ENROOT_CACHE_PATH          /home/ubuntu/enroot/cache
    ENROOT_DATA_PATH           /home/ubuntu/enroot/data
    ENROOT_TEMP_PATH           ${TMPDIR:-/tmp}
    ```
    
    Move contents of `/usr/lib/enroot` to new location: `/home/ubuntu/enroot/library`
    
    https://github.com/NVIDIA/enroot/issues/13
    
    https://github.com/NVIDIA/pyxis/issues/62
    
    https://github.com/NVIDIA/pyxis/issues/12

# Torchtitan over enroot + pyxis

## Enroot testing:

1. In compute nodes, build image with:

```bash
docker build -f torchtitan.dockerfile --build-arg HF_TOKEN="$HF_TOKEN" -t torchtitan_cuda128_torch27 .
```

2. [Import](https://github.com/NVIDIA/enroot/blob/master/doc/cmd/import.md) dockerd image to enroot (Can be done with docker://IMAGE:TAG from registry)

```bash
enroot import dockerd://torchtitan_cuda128_torch27
```

3. create enroot container
```bash
enroot create -n enroot_torchtitan /home/ubuntu/torchtitan_cuda128_torch27.sqsh
```

4. test it
```bash
enroot start enroot_torchtitan sh -c 'grep PRETTY /etc/os-release'
````
