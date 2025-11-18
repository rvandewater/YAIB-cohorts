# YAIB-cohorts — Docker quick start

Run the YAIB cohorts code in a reproducible Linux container. No prior Docker experience required.

## Prerequisites

### macOS

Install **Docker Desktop** _or_ install via Homebrew:

```bash
brew install docker docker-compose colima docker-buildx
# start the local Linux VM used by Docker CLI
colima start   
```

Note that if you encounter memory issues, you can increase Docker memory by using the `--memory` parameter. For example, this `colima start --memory 24` sets the memory to 24GB.

Verify:
```bash
docker --version
docker compose version
docker buildx version
```

If you use Docker Desktop on macOS, ensure it is running before you build.

## Clone the repo

```bash
git clone https://github.com/rvandewater/YAIB-cohorts.git
cd YAIB-cohorts
```

## Configure the data volume

The Compose file mounts two paths into the container:

* Project code → /home/ruser/app (mounted from the repo root, read/write).
* RICU datasets → /home/ruser/data (you must point this to your local dataset directory).

Edit docker/docker-compose.yml and set the datasets path:

```yaml
services:
  yaib_cohorts:
    volumes:
      - ..:/home/ruser/app                                      # repo root mounted into the container. DO NOT CHANGE.
      - [PATH_TO_RICU_DATA_ON_YOUR_MACHINE]:/home/ruser/data   # <-- change to the left of `:`
```

## Build the image

From the Docker subfolder:

```bash 
cd docker
docker compose build yaib_cohorts       
```

If you change the Dockerfile and need a clean rebuild:

```bash 
docker compose build --no-cache      
```

## Start an interactive shell

Run the service defined in docker-compose.yml:

```bash 
docker compose run --rm yaib_cohorts bash
```

You are now inside the container at `/home/ruser/app`.

Quick checks:

```bash 
pwd                      # should be /home/ruser/app
ls /home/ruser/data      # should show your RICU datasets (if you already imported any)
```

## Run the code

Navigate to `R` or `Python` and follow the READMEs there


## Exiting and cleaning up

Exit the shell: `exit`

Remove stopped containers built by Compose: `docker compose down --remove-orphans`

Remove the image if needed: `docker image rm yaib_cohorts_image`

Stop Colima: `colima stop`

