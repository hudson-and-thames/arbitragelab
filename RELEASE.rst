Version Release Guidelines
=======================

This document describes the guidelines for releasing new versions of the library. We follow semantic versioning, which means our version numbers have three parts: MAJOR.MINOR.PATCH.

- MAJOR version when you make incompatible API changes
- MINOR version when you add functionality in a backwards-compatible manner
- PATCH version when you make backwards-compatible bug fixes


1. Install the `bump2version` package:

    ```
    pip install --upgrade bump2version
    ```
--------------------

2.  Create a new branch for the release from dev branch:

    ```
    git checkout -b release/x.y.z
    ```
--------------------

3. Update the version number using the `bump2version` command:

    ```
    bump2version path
    ```
    or
    ```
    bump2version minor
    ```
    or
    ```
    bump2version major
    ```
--------------------

4. Commit the changes with the following message and push the changes to the release branch:

    ```
    git commit -m "Bump version: {current_version} → {new_version}"
    ```

    ```
    git push origin release/x.y.z
    ```

--------------------

5. Create a pull request from the release branch to the dev branch.

6. Once the pull request is approved and merged,  create the tag to invoke the development package publishing workflow on TestPypi.

        ```
        git tag -a x.y.z-dev -m "Release x.y.z-dev"
        ```

        ```
        git push origin tag x.y.z-dev
        ```

7. Test the development version on a new fresh environment and if everything is working fine, create a new pull request from the develop branch to the master branch.

8. Once the pull request is approved and merged, create the tag on the main branch to invoke the package publishing workflow:

    ```
    git tag -a x.y.z -m "Release x.y.z"
    ```

    ```
    git push origin tag <tag_name>
    ```
--------------------

9. Once the tag is pushed, the package publishing workflow will be triggered and the package will be published to the PyPI.

10. Once the package is published, create a new release on GitHub with the tag name and the release notes (generate them automatically).

11. Write a blog post announcing the release and publish it on the website.
12. Send a newsletter email announcing the release
13. Post on social media about the release
