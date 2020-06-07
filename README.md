# Setup instructions

During this course, you will be required to submit programming assignments
using the `git` version control system.
If you are not familiar with it, working through a tutorial is highly recommended:

As a start, the official user manual and an interactive walk-through can be found here:
- [https://git-scm.com/docs/user-manual.html](https://git-scm.com/docs/user-manual.html)
- [https://try.github.io](https://try.github.io)

## Forking

Forking this project will create a copy of the existing materials in this repository.

1. Please sign in to [https://git.uni-konstanz.de/](https://git.uni-konstanz.de/)
	 using your Shibboleth / Uni KN account
2. Fork this project: [https://git.uni-konstanz.de/hanhe.lin/dlp19/forks/new](https://git.uni-konstanz.de/hanhe.lin/dlp19/forks/new).
	 Please go to your project page now.
3. Setup the correct permissions:
	- Set your fork to `Private`: Settings -> General -> Permissions -> Project visibility
	- Grant `Master Access`: Settings -> Members -> Add Member to the following people:
		- @hanhe.lin

## Cloning

Please clone your forked repository to work on it using your local machine.

```shell
git clone git@git.uni-konstanz.de:FIRST.LAST/dlp.git
```
- Please copy the correct git@... URL from yor project website.
- Keep in mind that this (unlike e.g. SVN) creates a fully-fledged local repository!

## Tracking the Upstream Repository

We will regularly update the `dlp` repository throughout the semester.
Please add it as a remote repository to keep track of the changes:
```shell
git remote add -f tutorial git@git.uni-konstanz.de:hanhe.lin/dlp19.git
```
you can check that you have two remote repositories by running:
```shell
git remote -v
```
the output should list a fetch and a push URL for each repository:
```shell
origin	 git@git.uni-konstanz.de:FIRST.LAST/dlp.git (fetch)
origin	 git@git.uni-konstanz.de:FIRST.LAST/dlp.git (push)
tutorial git@git.uni-konstanz.de:hanhe.lin/dlp19.git (fetch)
tutorial git@git.uni-konstanz.de:hanhe.lin/dlp19.git (push)
```

## Merging

Merging allows you to conveniently copy new material from the tutorial repository
to your local repository:

```shell
git fetch --all
#[EXPECTED OUTPUT]:
#On branch master
#Your branch is up to date with 'origin/master'.
#
#nothing to commit, working tree clean

git merge tutorial/master
```

This step has to be repeated afer each release of new material.
- Resolve potential merge conflicts locally after each merge.
- Keep in mind that you do not have privileges to push to `tutorial/master`!

## Submitting Assignments

1. Commit and push your changes to the `master` branch of your GitLab repository.
	 You can of course use as many branches as you want, but we will only

2. `Tag` your final submission using either of the two methods:
  - GitLab Webinterface:
	  - Go to Repository -> Tags -> New
		- Choose the `master` branch to which you already (!) pushed your submission.
	- Git command line. Merge your submission to the `master` branch and check it out. Don't forget the
		--tags switch on the push command:
```shell
git checkout master
git tag -a exXY
git push
git push --tags
```
Replace XY with the exercise number you're currently submitting.


## Python Environment

In your repository, you find a file called `requirements.txt`.
It includes the version specifications for python packages that we will use
throughout this tutorial to ensure seamless compatibility with the continuous
integration system we're using to check your submissions.

### Windows Users:

Setting up all the required software can be a bit troublesome on Windows.
You might want to use a Linux VM or consider switching to a sane operating system altogether ;-)

If you desperately _want_ to use Windows and things didn't work for you out of the box,
here are a few suggestions on how to fix numpy/scipy installation issues that occured last year:

- Get the Linux for Windows Subsystem: [https://docs.microsoft.com/en-us/windows/wsl/install-win10](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
- Install python and pip for Windows and add python to PATH: [https://python-forum.io/Thread-Basic-Part-1-Python-3-6-and-pip-installation-under-Windows](https://python-forum.io/Thread-Basic-Part-1-Python-3-6-and-pip-installation-under-Windows)
- Important: `DO NOT use the space characer in directory names at any level!`

```shell
pip install virtualenv
virtualenv -p python .env
.env\Scripts\activate
pip install -r requirements_windows.txt
```
scipy and numpy can be troublesome - if `pip` fails to install them you can either try `easy_install`

```shell
easy_install numpy==1.14.2
easy_install scipy=0.19.1
```
if this also doesn't work, you can manually download precompiled versions of the libraries and install them using pip:

  - [https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy](https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
  - [http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)

```shell
pip install ..\numpy*.whl
pip install ..\scripy*.whl
```


### Linux:

```shell
pip install virtualenv            # Only required if you do not have virtualenv installed
virtualenv -p python3 .env
source .env/bin/activate          # Activate environment. [Usually prepends shell prompt with "(.env)"]
pip install -r requirements.txt   # Install dependencies locally for this environment

# work with stuff here:
cd src/00/
jupyter-notebook                  # if this does not open a browser window on your machine, navigate to localhost:8888

deactivate                        # Leave virtual environment
```

## Questions?

Just send me mail: hanhe.lin@uni.kn
