<div align="center">
   <a href="https://hudsonthames.org/arbitragelab">
   <img src="https://hudsonthames.org/wp-content/uploads/2021/04/featured-picture-arbitragelab.jpg" height="100%" 
   style="margin-left: auto; margin-right: auto; display:block;">
   
   </a>
  </br>
</div>

# Welcome to Statistical Arbitrage Laboratory! 

## What is ArbitrageLab?

ArbitrageLab is a python library that enables traders who want to exploit mean-reverting portfolios
by providing a complete set of algorithms from the best academic journals.


## Development

### Creating a release

- Create `release/<version>` branch
- Bump versions throughout source files (we use `bump2version` to do automate this process, TODO: Add instructions)
- Update customer install instructions in documentation source files
- Update release information in changelog in documentation source files
- Open PR from `release` branch into `develop`
- Merge PR once approved
- Obfuscate `develop` using PyArmor (instructions are located elsewhere in this README)
- Test you can install the wheel from a fresh environment
- Merge `develop` into `master`
- Upload the obfuscated wheel to the Hudson & Thames Clients organization
- Tag the commit with the version number
- Write a blog post announcing the release
- Send a newsletter email
- Post on social media

### Bumping version numbers using `bump2version`

We use `bump2version` to automatically bump versions throughout source files.

Configuration lives in the `.bumpversion.cfg` file. To run `bump2version`, first install it via `pip`:

``` sh
pip install --upgrade bump2version
```

And then bump the version:

``` sh
bump2version <version-bump-type>
```

where `<version-bump-type>` tells you which version to be bumped. The acceptable
values are `major`, `minor` or `patch`, conforming to the semantic versioning
pattern: `major.minor.patch`. For example, `3.2.7` has a major version of 3, a
minor version of 2 and a patch version of 7.

