import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='rge',  
     version='0.4',
     author="Ron Arel",
     scripts=['rge_pkg/rge'] ,
     author_email="RonArel123@gmail.com",
     description="Engine for Refrence Game Tests",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/cocolab-projects/reference-game-exploration",
     packages=setuptools.find_packages("src"),
     package_dir={"": "src"},
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
