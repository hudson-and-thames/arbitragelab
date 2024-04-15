# Contribution Guidelines for Hudson and Thames ArbitrageLab

Thank you for considering contributing to Hudson and Thames ArbitrageLab! We value your time and effort in helping
improve our project. Please follow these guidelines to ensure a smooth contribution process for everyone involved.

## Getting Started

Before you begin:
- Make sure you have a GitHub account.
- Familiarize yourself with our project by reading the documentation [here](https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/).
- Check the issues page for outstanding work and discussions to see if someone else is already working on your idea.

## Making Contributions

To contribute to our project, follow these steps:

1. **Fork the Repository**
   - Go to the GitHub page of Hudson and Thames ArbitrageLab and click the "Fork" button at the top right corner.

2. **Clone Your Fork**
   - After forking, clone the repository to your local machine to start making changes:
     ```bash
     git clone https://github.com/your_username/arbitragelab.git
     cd arbitragelab
     ```

3. **Create a Branch**
   - Create a new branch for your changes:
     ```bash
     git checkout -b feature/your_feature_name
     ```

4. **Make Changes Locally**
   - Implement your feature or bug fix.
   - Write clear, comprehensible commit messages.
   - Make sure your code adheres to the existing style of the project to maintain its readability.

5. **Coverage and Unit Test**
   - Run all unit tests to confirm your changes don't break existing functionality. We require 100% coverage.
   - Check code coverage and improve it if possible. Coverage reports should be part of the project's test suite.

6. **Follow PR Template**
   - When creating a pull request, ensure you follow one of our PR templates provided in the repository. This helps maintain the project's consistency and facilitates review.

7. **Submit a Pull Request**
   - Push your changes to your fork:
     ```bash
     git push origin feature/your_feature_name
     ```
   - Go to the repository on GitHub, and you'll see a "Compare & pull request" button. Click it and fill in the details according to the chosen template.
   - Submit the pull request for review, to the develop branch - never master/main.

## Code Review

Your pull request will be reviewed by maintainers who may provide feedback or request changes. Keep an eye on your GitHub notifications and respond promptly to comments.

## Acceptance Criteria

Before a pull request is accepted, it must:
- Pass all automated build checks.
- Achieve successful results on all unit tests.
- Maintain or improve existing code coverage.
- Adhere to the coding standards and documentation style of the project.

## Final Steps

Once your pull request is approved and merged, you are officially a contributor. Congratulations! We encourage you to 
continue participating in our community and consider tackling other issues or improving documentation.

For any questions or help with getting started, don't hesitate to reach out through our community forums or issue tracker.

Thank you for contributing to Hudson and Thames ArbitrageLab!
