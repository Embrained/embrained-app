# Contributing to Embrained

First off, thank you for considering contributing to Embrained! It's people like you that make Embrained an amazing open-source neurorobotics platform. We welcome contributions from everyone.

## Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check our [Issues](https://github.com/Embrained/embrained-app/issues) first to see if someone else has already created a ticket. If not, go ahead and [make one](https://github.com/Embrained/embrained-app/issues/new/choose)!

## How to contribute

1. **Fork the Repository:** Start by forking the `Embrained/embrained-app` repository.
2. **Clone your Fork:** Clone the forked repository to your local machine.
3. **Create a Branch:** Create a new branch for your feature or bug fix (`git checkout -b feature/my-new-feature` or `bugfix/issue-123`).
4. **Make Changes:** Make your changes, testing them locally. If you're modifying hardware communication, please test with your Plexus robot if possible.
5. **Commit:** Commit your changes with a clear and descriptive commit message.
6. **Push:** Push your branch to your fork on GitHub.
7. **Pull Request:** Open a Pull Request from your fork to the main `Embrained/embrained-app` repository. Provide a thorough description of the changes using the Pull Request template.

## Developing Locally
Please refer to the `README.md` for instructions on using `setup.bat`/`setup.sh` to configure your environment. When adding new Python dependencies, please add them to `requirements.txt`. For frontend dependencies, ensure you run `npm install` and `npm run build` within the `cognitive-engine/frontend` directory before submitting a PR.

## Code Style
- For Python code, try to follow PEP 8 conventions.
- For React/Frontend, ensure your components are clean and use Tailwind safely. 
- Please include docstrings for any new python functions, especially those interacting with the Cognitive Engine or hardware.

## Data Sharing
Since data collection is the core bottleneck we are trying to solve, we highly encourage sharing your datasets, models, and weights! Please join our [Discord Community](https://discord.com/channels/1487132795833684228/1487132796920004640) to share datasets or trained `.pth` models, as GitHub is not ideal for hosting large binary files.

Thank you!
The Embrained Team
