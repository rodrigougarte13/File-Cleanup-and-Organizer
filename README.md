# File Cleanup and Organizer

This application was developed to streamline organization and cleanliness within a folder, initially targeting my cluttered downloads folder. The first iteration (Cleaner.py) was a basic tool that sorted files by their types. As the project evolved, machine learning techniques were integrated to enhance functionality. Specifically, KMeans clustering (Cleaner with Clustering.py) was implemented to categorize files based on both type and name.

For my specific use case, this approach yielded highly beneficial results, effectively grouping files in a logical and intuitive manner. Users are encouraged to explore and adjust parameters within the app to potentially optimize clustering outcomes for their own folders.

To set up the environment, install these libraries using pip:
```python
pip install pandas scikit-learn
