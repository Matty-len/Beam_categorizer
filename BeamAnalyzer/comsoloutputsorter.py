import pandas as pd
import matplotlib.pyplot as plt
import ast
from collections import OrderedDict
class ComsolOutputSorter():
    def __init__(self) -> None:
        """This object is intended to read data from COMSOL multiphysics and make it into an pandas dataframe
        NOTE This object can be easily switched if on choose to export comsol data in another format.
        """
        pass
    
    def process_data_file(self) -> None:
        """Method to do all the substeps"""
        self.get_all_titles_and_title_indeces()
        self.count_all_eigenvalues()
        self.make_data_to_dict()
        self.make_dataframe_based_on_eigenvalues()
        self.add_effective_masses()

    def add_effective_masses(self) -> None:
        """Adds an effective mass `eff mass` column to the dataframe"""
        self.data['eff mass'] = list(self.mass_frame['mass'])

    def load_file_with_effective_masses(self, filename: str = None) -> None:
        """Method to load the effective masses of the eigenvalues, only given a filename, for example `thick_effective_mass.txt`
        The file is supposed to by of the csv type."""
        df = pd.read_csv(filename, delim_whitespace=True)
        self.mass_frame = df

    def load_file(self, filename: str) -> None:
        """Just opens the file object and get the lines for later manipulation"""
        with open(filename) as file:
            self.lines = file.readlines()
        file.close()

    def set_attributes(self, attributes: list = []) -> None:
        """Method for setting the attributes that should be included in the dataframe"""
        self.attributes = attributes

    def get_all_titles_and_title_indeces(self) -> None:
        """A method for reading all titles and indeces of titles, meaning the linenumber they appear"""
        titles = list()
        title_indeces = list()

        #Reading lines one by one, and if they have % they are marked as a heading.
        for index, line in enumerate(self.lines):
            if line[0] == '%':
                titles.append(line)
                title_indeces.append(index)
        # adding the titles into the object so we know what to look for when sort the data
        self.titles = titles
        self.titles_indeces = title_indeces

    def count_all_eigenvalues(self) -> None:
        """Method to count for how many eigenvalues there have been generated and generate a list of the for the table"""
        count = 0
        for line in self.lines:
            #Use lambda AND attribute 0 so we do not count the number of attribute times
            if 'lambda' in line and self.attributes[0] in line:
                count += 1
        self.count = count
        self.eigenvalues = range(1, count +1) # Make a range of eigenvalues for later use.

    def make_data_to_dict(self) -> None: # Had to be ugly because of the ugly structure of comsol outputs!
        """This method is to convert the lines we have read to useful data, it does so by reading a sequence of the file
        from one title to the next and take the data in between."""
        dictionary = dict()
        for internal_index, index in enumerate(self.titles_indeces): # index of the lines to check that we have not gone too long
            if internal_index == len(self.titles_indeces) - 1:
                break
            # 1. We get the indeces from where the info lines start to where they end, thats from one title to another.
            this_title_index = self.titles_indeces[internal_index]
            next_title_index = self.titles_indeces[internal_index+1]
            # We then take all the lines with numbers from this range and put them into an array, we check that they are Data attributes only
            # We add them to the dictionary with their respective titles.
            dictionary[self.titles[internal_index]] = [float(line) for line in self.lines[this_title_index+1:next_title_index] if 'Data' in self.titles[internal_index]]
        
        # Special case is the last line to the bottom, but do the same again and add to dictionary
        last_line = len(self.lines)
        last_title = self.titles_indeces[-1] # the line number where last title is
        dictionary[self.lines[last_title]] = [float(x) for x in self.lines[last_title+1:last_line]]
        self.data_dict = dictionary
       
    def make_dataframe_based_on_eigenvalues(self) -> None:
        """This method takes the `dictionary` and converts it into a `dataframe`, which can be more easy to work with."""
        full_frame = pd.DataFrame()
        for eigenvalue in self.eigenvalues: # for all eigenvalues 
            dc = OrderedDict()
            for attribute in self.attributes:
                for title in self.data_dict.keys(): # we search all titles
                     #if we can find our attribute string in the title and the eigenvalue is correct then we want to rename our attribute list
                    if attribute in title and f'lambda={eigenvalue})' in title.split():
                            # row is lambda=eigenvalue
                            # column is attribute
                            dc[attribute] = str(self.data_dict[title])
            # Pandas can natively make a dataframe from a dictionary.
            df = pd.DataFrame(data=dc, index=[eigenvalue])
            # Each of the subframes gets added to the fulle frame
            full_frame = pd.concat([full_frame, df])
        # making the full datafram into what we call data
        self.data = full_frame

    def extract_feature(self, attribute: str, eigenvalue: int) -> list:
        """Extracts the data in the dataframe into a list for quick inspections and plotting."""
        return ast.literal_eval(self.data[attribute][eigenvalue])

# Now instead we set the thing up to make the cut lines, and then just ignore everything else.

