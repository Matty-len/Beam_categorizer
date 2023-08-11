from BeamAnalyzer.modeclassifier import ModeClassifier
class ClasificationComparer():
    def __init__(self, class1: ModeClassifier, class2: ModeClassifier) -> None:
        self.class1 = class1
        self.class2 = class2

    def print_where_modes_disagree(self):
        iterables = (self.class1.predicted_modes, self.class1.data.eigenvalues, self.class1.frequencies, self.class2.predicted_modes)
        for mode1, mode_num, freq, mode2 in zip(*iterables):
            if mode1 != mode2:
                print(f'{mode_num:<20} {freq:<20.4e} {mode1:<20} {mode2:<20} Disagree')
                
    def do_2_classificaitons_agree(self):
        iterables = (self.class1.predicted_modes, self.class1.data.eigenvalues, self.class1.frequencies, self.class2.predicted_modes)
        for mode1, mode_num, freq, mode2 in zip(*iterables):
            if mode1 == mode2:
                print(f'{mode_num:<20} {freq:<20.4e} {mode1:<20} {mode2:<20} Agree')
            else:
                print(f'{mode_num:<20} {freq:<20.4e} {mode1:<20} {mode2:<20} Disagree')
