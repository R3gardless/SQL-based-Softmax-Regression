# SQL based Softmax Regression

## Intro

Repository for __*Implementing SQL-based Softmax Regression for MNIST data*__ paper experiment code [[paper](https://drive.google.com/file/d/18ymVk7ZeyMQxpIw3iB-CqTMkhwqKSuWq/view?usp=drive_link)] [[ppt](https://drive.google.com/file/d/1S6TcjOeh1CW-vy_LbuR_VzrFgl90l8RN/view?usp=sharing)]

## Requirements

- **Ubuntu** 18.04.6 LTS
- **Anaconda** 23.7.2 
- **PostgreSQL** 16.0

## Installation

### Python (Hyper API)

1. Install using `environment.yml` file
```bash
$ conda env create -f environment.yml
```
2. Activate conda environment
```bash
$ conda activate sql-4-ml
```

3. If you want to delete conda environment
```bash
$ conda env remove -n sql-4-ml
```

### PostgreSQL

1. Create a user and login to postgres account
```bash
$ adduser postgres
$ su - postgres
```

2. Download the source code from PostgreSQL [website](https://www.postgresql.org/ftp/source/)
```bash
$ wget https://ftp.postgresql.org/pub/source/v16.0/postgresql-16.0.tar.gz
$ tar -xvzf postgresql-16.0.tar.gz
```
3. Make a directory for build and configure the source tree
```bash
$ cd postgresql-16.0
$ mkdir build
$ ./configure --prefix=/home/postgres/postgrsql-16.0/build
```

4. Build and install the source code (make -j = Make build faster using multiple processes)
```bash
$ make -j install
```

5. Add the shared library path to `~/.bashrc`
```bash
$ vim ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/postgres/postgresql-16.0/build/lib
export PATH=/home/postgres/postgresql-16.0/build/bin:$PATH

$ source ~/.bashrc
```
- Add the shared library to `.profile` of postgres user
```bash
$ cd ~
$ vim .profile
PATH=$PATH:/home/postgres/postgresql-9.4.5/build/bin
export PATH
$ . ~/.profile
```
6. Initialize the database storage with `initdb` command of postgres
```bash
$ initdb -D /home/postgres/test_data
```

## How to Execute?

### Run PostgreSQL

```bash
$ pg_ctl -D /home/postgres/test_data -l logfile start
```
- Stop the PostgreSQL server
```bash
$ pg_ctl -D /home/postgres/test_data -m smart stop
```
### Run Experiment 
```bash
$ conda activate sql-4-ml
$ python3 experiment.py
```


## References
- https://github.com/mark-blacher/sql-algorithms
- https://github.com/LeeBohyun/postgreSQL/blob/main/postgres-installation-guide.md

