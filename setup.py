import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chatbot-creator",
    version="0.0.3",
    author="Pranava Mohan",
    license="MIT",
    description="Python package for creating chatbots.(including DiscordBot).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/My-refer/chatbot-creator",
    packages=['ChatbotCreator'],
    install_requires=["pandas", "keras", "sklearn", "numpy", "nltk", "spacy", "tensorflow", "discord", "google", "bs4"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
