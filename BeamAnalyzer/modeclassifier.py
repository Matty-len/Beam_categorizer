from scipy.stats import skewtest
from math import log10
import numpy as np
import os
from sklearn.metrics import r2_score
from scipy.integrate import simpson
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from BeamAnalyzer.comsoloutputsorter import ComsolOutputSorter


class ModeClassifier():
    """
    3 Types of things we want, L = longi, F = flexural, T = torsional, U strings
    
    """
    def __init__(self, comsolsorted_data: ComsolOutputSorter, folder_name='default') -> None:
        """ModeClassifier uses simple assesment on the form of the data to classify if the modes are one of the following
        L = longitudinal
        F = Flexural
        T = Torsional
        U = Unknown
        """
        self.data = comsolsorted_data
        self.foldername = folder_name
        
        # Configurations
        self.save_vizu = True
        self.visualise_mode_plots = True
        self.line_expression = "bottom-(d_top/2)" # default line expressions, should be assesed each time
        self.predicted_modes = None
        self.frequencies = list()
        self.deflection_angle = list()
        self.paired_mode = list()
        self.categorized_modes = {'x_PureFlexural':[],'y_PureFlexural':[], 'PureTorsional':[], 'UnknownCategory': [],'FlexuralTorsional': [], 'PureLongitudinal': []}
        self.mode_pairs = []
        self.dominant_direction = []
        self.passed_eigenfreq_closeness = []


    def plot_mode_spacing(self) -> None:
        """Plots the modes versus the frequency of the modes"""
        modes = []
        freqs = []
        # Collect a list of modes placed at 1 and all the freqs
        for mode in self.data.eigenvalues:
            modes.append(1)
            # collect one freq for each eigenvalue, since all the freqs are identical
            freqs.append(np.asarray(self.data.extract_feature(attribute='freq', eigenvalue=mode))[0])
        plt.plot(modes, freqs)

    def make_folder_structures(self)->None:
        """Makes the folder for where the different modes go into. It creates a folder with the same name¨
        as the filename we look into and also subfoldersw"""
        if self.save_vizu:
            if not os.path.exists(self.foldername):
                os.makedirs(self.foldername)
            if not os.path.exists(f'{self.foldername}/slope_test'):
                os.makedirs(f'{self.foldername}/slope_test')
            if not os.path.exists(f'{self.foldername}/R2_test_poly'):
                os.makedirs(f'{self.foldername}/R2_test_poly')
            if not os.path.exists(f'{self.foldername}/R2_test_linear'):
                os.makedirs(f'{self.foldername}/R2_test_linear')
        if True:
            for mode_type in self.categorized_modes.keys():
                if not os.path.exists(f'{self.foldername}/{mode_type}'):
                    os.makedirs(f'{self.foldername}/{mode_type}')
                    
    def classify_all_modes(self)->None:
        """Clasiifies all modes with the approaches defined in classfiy mode. Saves the predicted modes
        into self.predicted modes, which can be accesed afterwards to see what modes the vibrations belong to."""
        self.make_folder_structures()
        self.predicted_modes = []
        if self.save_vizu == True:
            print("WARNING: It is slow to plot, you can set save_vizu == False, to increment speed")
        for mode in self.data.eigenvalues:
            modetype = self.classify_mode(mode)
            self.predicted_modes.append(modetype)
        #At last we also wanna pair our predicted modes
        self.pair_modes()
        
    def print_all_predictions(self)-> None:
        """Prints all the predictions with modnumber, frequency, prediction, deflection angle, its pair and dominant direction."""
        print(len(self.paired_mode))
        for mode in self.data.eigenvalues:
            print(f'{mode:<10} {self.frequencies[mode-1]:<10.4e}  {self.predicted_modes[mode-1]:<20} {self.deflection_angle[mode-1]:<10.2f} {self.paired_mode[mode-1]:<10} {self.dominant_direction[mode-1]:<10}')

        
    def export_datasheet(self) -> None:
        for modetype in self.categorized_modes:
            modes = np.asarray(self.predicted_modes)
            modenumbers = np.asarray(self.data.eigenvalues)
            
            with open(f'{self.foldername}/{modetype}/{modetype}_results.txt', 'w') as f:
                f.write(f'modenumbers: \t\t {str(list(modenumbers[modes == modetype]))}\n\n')
                f.write(f'freq: \t\t {str(list(np.asarray(self.frequencies)[modes == modetype]))}\n\n')
                # creating things
                
                u_vec = list()
                v_vec = list()
                w_vec = list()
                mass_vec = list()
                for mode_num in modenumbers[modes == modetype]:
                    umax = np.asarray(self.data.extract_feature(attribute='maxop1(u)', eigenvalue=mode_num))[0]
                    vmax = np.asarray(self.data.extract_feature(attribute='maxop1(v)', eigenvalue=mode_num))[0]
                    wmax = np.asarray(self.data.extract_feature(attribute='maxop1(w)', eigenvalue=mode_num))[0]
                    mass = list(self.data.data.loc[:,"eff mass"])[mode_num-1]
                    u_vec.append(umax)
                    v_vec.append(vmax)
                    w_vec.append(wmax)
                    mass_vec.append(mass)
                f.write(f'mass:  \t\t {str(list(mass_vec))}\n\n')
                f.write(f'maxop1(u): \t\t {str(u_vec)}\n\n')
                f.write(f'maxop1(v): \t\t {str(v_vec)}\n\n')
                f.write(f'maxop1(w): \t\t {str(w_vec)}\n\n')

                for attribute in ['solid.eZZ', 'solid.eXX', 'solid.eYY', 'solid.eXZ', 'solid.eYZ', 'solid.eXY']:
                    all_eZZ_mat = []
                    eZZ_pr_vec = list()
                    eZZ_mr_vec = list()
                    eZZ_0_vec = list()
                    for mode_num in modenumbers[modes == modetype]: # hmm ikke dem allesammen, kun dem som har den ri
                        all_pos = np.asarray(self.data.extract_feature(self.line_expression, mode_num))
                        all_eZZ = np.asarray(self.data.extract_feature(attribute, mode_num))
                        all_eZZ_mat.append(list(all_eZZ))
                        eZZ_pr = all_eZZ[-1]
                        eZZ_pr_vec.append(eZZ_pr)
                        eZZ_mr = all_eZZ[0]
                        eZZ_mr_vec.append(eZZ_mr)

                        eZZ_0 = all_eZZ[all_pos == find_nearest(all_pos, 0)][0]
                        eZZ_0_vec.append(eZZ_0)
                    
                    f.write(f'+0 {attribute}: \t\t {str(list(eZZ_0_vec))}\n\n')
                    f.write(f'-R {attribute}: \t\t {str(list(eZZ_mr_vec))}\n\n')
                    f.write(f'+R {attribute}: \t\t {str(list(eZZ_pr_vec))}\n\n')


                #f.write(f'all eZZ: \t\t {str(list(all_eZZ_mat))}\n\n')

                #effective mass


    def classify_mode(self, mode:int) -> str:
            """Classifies one mode by analysing the strain components given the eigenvalue/mode `int`
            The method does both go through assertion as well as if the mode is and x or y mode for the flexural.
            The flow of the code is as follows.
            1. Find the classification of pure and shear strain.
            2. make the frequency in the frequencies list.
            3. Find the deflection angle.
            4. go through a controlflow on the behavior of what modes belong where.
            5. On the flexural modes give them a label of x and y.
            """

            #1. Find the classification of pure and shear strain
            pure_comp = self.analyse_pure_strain(mode=mode)
            shear_comp = self.analyse_shear_strain(mode=mode)
            #2. make the frequency in the frequencies list
            freq = np.asarray(self.data.extract_feature(attribute='freq', eigenvalue=mode))[0]
            self.frequencies.append(freq)
            
            #3. Find the deflection angle.
            umax = np.asarray(self.data.extract_feature(attribute='maxop1(u)', eigenvalue=mode))[0]
            vmax = np.asarray(self.data.extract_feature(attribute='maxop1(v)', eigenvalue=mode))[0]
            self.deflection_angle.append(np.rad2deg(np.arctan(vmax/umax)))
            
            # pure_comp options: L, F, U
            # shear_comp options: T,F,U
            # Table
            # L = PureLongitudinal
            # FT = FlexuralTorsional
            # FF = PureFlexural
            # FU = PureFlexural
            # UT = PureTorsional
            # UF = UnknownCategory
            # UU = UnknownCategory

            #4. go through a controlflow on the behavior of what modes belong where.
            if pure_comp == "L": #TODO maybe more assertion
                cat_type = "PureLongitudinal"
            elif pure_comp == "F" and shear_comp == "T":
                cat_type = "FlexuralTorsional"
            elif pure_comp == "F" and shear_comp == "U":
                cat_type = "PureFlexural"
            elif pure_comp == "F" and shear_comp == "F":
                cat_type = "PureFlexural"
            elif pure_comp == "U" and shear_comp == "F":
                if self.is_data_antisymmetric(mode=mode, attribute='solid.eZZ'):
                    cat_type = 'PureFlexural'
                else:
                    cat_type = "UnknownCategory"
            elif pure_comp == "U" and shear_comp == "T":
                cat_type = "PureTorsional"
            elif pure_comp == "U" and shear_comp == "U":
                if self.is_data_antisymmetric(mode=mode, attribute='solid.eZZ'):
                    cat_type = 'PureFlexural'
                else:
                    cat_type = "UnknownCategory"
            else:
                cat_type = "UnknownCategory"

            # 5. On the flexural modes give them a label of x and y
            if cat_type == "PureFlexural":
                if self.deflection_angle[-1] > 0: #NOTE this might not be good enough, should have used a mix of cos and sine
                    self.dominant_direction.append('x')
                    cat_type = "x_PureFlexural"
                elif self.deflection_angle[-1] < 0: 
                    self.dominant_direction.append('y')
                    cat_type = "y_PureFlexural"
            else:
                self.dominant_direction.append('-')
            
            if self.visualise_mode_plots:
                self.do_visualise_mode_plots(mode=mode, cat_type=cat_type, pure_comp=pure_comp, freq=freq, shear_comp=shear_comp)
            return cat_type 
    
    def do_visualise_mode_plots(self, mode: int, cat_type: str, pure_comp: str, freq: str, shear_comp: str)-> None:
        """Visualizes the pure strain and deflections on graphs, giving essentially the data we use for assertion
        but just on plot so we can control that we agree on the machines decisions."""
        axis_0 = np.asarray(self.data.extract_feature(attribute=self.line_expression, eigenvalue=mode))
        axis_pure = np.asarray(self.data.extract_feature(attribute='solid.eZZ', eigenvalue=mode))
        u = np.asarray(self.data.extract_feature(attribute='u', eigenvalue=mode))
        v = np.asarray(self.data.extract_feature(attribute='v', eigenvalue=mode))
        w = np.asarray(self.data.extract_feature(attribute='w', eigenvalue=mode))
        dominant_shear = self.find_dominant_shear_strain_feature(mode=mode)
        parabolic_like, scorep = self.is_data_parabolic_like(mode = mode)
        linear_like, scorel = self.is_data_linear_like(mode=mode)
        fig, axs = plt.subplots(1, 3)
        axs[0].plot(axis_0, axis_pure)
        axs[0].plot(axis_0[0], axis_pure[0], 'o', color='r')
        axs[0].plot(axis_0[-1], axis_pure[-1], 'o', color='r')
        axs[0].plot(axis_0[axis_0 == find_nearest(axis_0, 0)], axis_pure[axis_0 == find_nearest(axis_0, 0)], 'o', color='r')
        axs[0].set_ylabel('solid.eZZ')
        axs[0].set_xlabel(self.line_expression)
        fig.suptitle(f'mode:{mode}, cat:{pure_comp}, freq:{freq:.4e}, pure:{pure_comp}, shear:{shear_comp}, par: {parabolic_like}, s:{round(scorep,4)} - lin: {linear_like}, s: {round(scorel, 4)}', fontsize = 9)
        
        XY = np.asarray(self.data.extract_feature(attribute='solid.eXY', eigenvalue=mode))
        XZ = np.asarray(self.data.extract_feature(attribute='solid.eXZ', eigenvalue=mode))
        YZ = np.asarray(self.data.extract_feature(attribute='solid.eYZ', eigenvalue=mode))
        axs[1].plot(axis_0, XY, label = 'xy')
        axs[1].plot(axis_0, XZ, label  = 'xz')
        axs[1].plot(axis_0, YZ, label = 'yz')
        axs[1].set_xlabel(self.line_expression)
        axs[1].set_ylabel(dominant_shear)
        axs[1].legend()

        axs[2].plot(axis_0, u, label='u')
        axs[2].plot(axis_0, v, label='v')
        axs[2].plot(axis_0, w, label='w')
        axs[2].set_xlabel(self.line_expression)
        axs[2].set_ylabel('displacement')
        axs[2].legend()
        plt.savefig(f"{self.foldername}/{cat_type}/{mode}.png")
        plt.close()
         
    
    def analyse_pure_strain(self, mode: int) -> str:
            """Method that only considers the ZZ component of the strain, the method checks the following
            1. Does the data have the same sign everywhere (either negative or positive)?
            2. Does the slope of the data have the same sign on both sides? NOTE: This was not implemented as it did not result in better analasys
            3. How parabolic like is the data? (default r_score = 0.6) NOTE: This did not give better insight so it is outcommented, but one can play with it
            4. How linear like is the data? (default r_score = 0.6)

            Parameters:
            -----------
            mode : `int`, is the number of mode for example mode number 1 or 2 or so on.
            """
            #0. Init
            midpoint = 0
            all_positive = False
            all_negative = False
            same_slope = False
            freq = np.asarray(self.data.extract_feature(attribute="freq", eigenvalue=mode))[0]

            #1. Same sign?
            if self.is_all_data_positive(modenumber=mode, attribute='solid.eZZ'):
                all_positive = True

            if self.is_all_data_negative(modenumber=mode, attribute='solid.eZZ'):
                all_negative = True
            same_sign_for_all = all_negative or all_positive
            
            ##2. Slope have same sign?
            # if self.does_data_have_same_slope_on_both_side_of_middle_point(mode=mode):    
            #     same_slope = True
            ##3. is data parabolic like? ##NOTE Taken out because data could be symmetric without being parabolic!
            #parabolic_like, score = self.is_data_parabolic_like(mode = mode)
            
            #4. Is data linear like?
            linear_like, score = self.is_data_linear_like(mode=mode)
            
            # criterions for Longitudinal categorization!
            if same_sign_for_all: #and parabolic_like and not same_slope and not linear_like:
                return "L"
            # criterions for Flexural mode
            if linear_like:
                return "F"
            #If we are not L or F this method votes for U = Unknown.
            return "U"     
    
    def analyse_shear_strain(self, mode: int) -> str:
        """Analyses the shear strain for a mode and votes if it sees Torsional/Flexural/Unknown features.
        Shear strain is defined as XY, YZ, XZ.
        Takes mode `int` as input to see what modenumber we should use.
        The method goes through the following steps.
        1. It figures out what the "dominant" feature is. It does so by figuring out which of the 3 component have the maximal amplitude.
        2. Is the data linear like on dominant feature? (Have R score higher than 0.6 by default)
        3. Is the data parabolic like on dominant feature? (Have R score higher than 0.6 by default).
        4. Vote for the mode the fits in the shear strain domain.
        """
        #1/2/3. checking for dominant feature and goes through the criterion
        dominant_shear_feature = self.find_dominant_shear_strain_feature(mode=mode)
        linear_like, score_linear = self.is_data_linear_like(mode=mode,attribute=dominant_shear_feature)
        parabolic_like, score_parabolic = self.is_data_parabolic_like(mode=mode, attribute=dominant_shear_feature)

        #4. Voting from criterions.
        if linear_like and not parabolic_like:
            return "T"
        if parabolic_like and not linear_like:
            return "F"
        return "U"
 
    def find_dominant_shear_strain_feature(self, mode: int) -> str:
        """Figures out which shear strain is the most dominant by asserting the amplitude, on the mode `ìnt` which is the mode number.
        1. Extracts the curves from XY, XZ, YZ
        2. Check which of the components have the max amplitude"""
        #1. Extracting curves #NOTE eXY was in this framwork more confusing to treat and the framework did not have the means
        # of handling it, so it is ignored for now.
        #XY = np.asarray(self.data.extract_feature(attribute='solid.eXY', eigenvalue=mode))
        XZ = np.asarray(self.data.extract_feature(attribute='solid.eXZ', eigenvalue=mode))
        YZ = np.asarray(self.data.extract_feature(attribute='solid.eYZ', eigenvalue=mode))
        
        #2. Check for the max amplitude on the components.
        # XY = max(abs(XY))
        XZ = max(abs(XZ))
        YZ = max(abs(YZ))
        check_list = [XZ, YZ]
        # if max([XY, XZ, YZ]) == XY:
        #     return 'solid.eXY'
        #3. Returns the dominant component.
        if max(check_list) == XZ:
            return 'solid.eXZ'
        if max(check_list) == YZ:
            return 'solid.eYZ'
        raise ArithmeticError("We should not be down here")

    def is_all_data_positive(self, modenumber: int, attribute: str) -> bool: #vague indication of symmetry
        """Extracts the data belonging to and attribute `str` (as named in COMSOL) for a mode number `int` and checks that all the data points are below 0."""
        data_quantity = self.data.extract_feature(attribute=attribute, eigenvalue=modenumber)
        for data_point in data_quantity:
            if data_point < 0:
                return False
        return True

    def is_all_data_negative(self, modenumber: int, attribute: str) -> bool: #vague indication of symmetry
        """Extracts the data belonging to and attribute `str` (as named in COMSOL) for a mode number `int` and checks that all the data points are above 0."""
        data_quantity = self.data.extract_feature(attribute=attribute, eigenvalue=modenumber)
        for data_point in data_quantity:
            if data_point > 0:
                return False
        return True

    def plot_order_of_magnitude(self) -> None:
        """Plots the logarithm of themaximal datapoint for the ZZ direction, versus the eigenvlaues
        Mostly obsolete
        """
        maxvals = []
        for mode in self.data.eigenvalues:
            maxval = log10(abs(max(self.data.extract_feature(attribute="solid.eZZ", eigenvalue=mode))))
            maxvals.append(maxval)
        
        plt.plot(self.data.eigenvalues, maxvals, 'o')
        plt.grid()
        plt.close()


    def is_data_linear_like(self, midpoint: float = 0, mode: int = None, r_score_limit: float = 0.6, attribute: str = 'solid.eZZ')-> (bool, float):
        """The method extracts the line expression, which is the expresions used in COMSOL to define where on the structure we are.
        The it takes and attribute, by defaul the ZZ strain. It figures out if the data have the same sign on the slope on both sides
        of the middle point.
        If this is true it figures out the best linear fit. and returns if the function is linear like (r_score better than r_score_limit) and 
        how much linear like it is
        """
        score = 0
        axis_0 = np.asarray(self.data.extract_feature(attribute=self.line_expression, eigenvalue=mode))
        axis_1 = np.asarray(self.data.extract_feature(attribute=attribute, eigenvalue=mode))
        freq = np.asarray(self.data.extract_feature(attribute="freq", eigenvalue=mode))[0]
        if self.does_data_have_same_slope_on_both_side_of_middle_point(midpoint=midpoint,attribute=attribute, mode=mode):
            #linear fit
            slope, cut = np.polyfit(axis_0,axis_1, 1)
            fit = (axis_0)*slope + cut
            score = r2_score(y_true=axis_1, y_pred=fit)
            if self.save_vizu == True:
                plt.figure()
                plt.plot(axis_0, axis_1)
                plt.plot(axis_0, fit)
                plt.xlabel(self.line_expression)
                plt.ylabel(attribute)
                plt.title(f'R^2 = {score}, \omega = {freq:.4e}')
                plt.savefig(fname = f"{self.foldername}/R2_test_linear/{mode}_{attribute}.png")
                plt.close()
            if score > r_score_limit:
                return True, score
        return False, score
    
    def is_data_parabolic_like(self, midpoint:float = 0, mode:int = None, r_score_limit:float = 0.6, attribute:str = 'solid.eZZ') -> (bool, float):
        """The method extracts the line expression, which is the expresions used in COMSOL to define where on the structure we are.
        The it takes and attribute, by default the ZZ strain. It figures out if the data have the same sign on the slope on both sides
        of the middle point.
        If this is true it figures out the best parabolic fit. and returns if the function is parabolic like (r_score better than r_score_limit) and 
        how much linear like it is
        """
        axis_0 = np.asarray(self.data.extract_feature(attribute=self.line_expression, eigenvalue=mode))
        axis_1 = np.asarray(self.data.extract_feature(attribute=attribute, eigenvalue=mode))
        freq = np.asarray(self.data.extract_feature(attribute="freq", eigenvalue=mode))[0]
        score = 0
        if not self.does_data_have_same_slope_on_both_side_of_middle_point(midpoint=midpoint, attribute=attribute, mode=mode):
            #parabolic fit on both sides.
            curvature, slope, cut = np.polyfit(axis_0,axis_1, 2)
            fit = (axis_0**2)*curvature + (axis_0)*slope + cut
            score = r2_score(y_true=axis_1, y_pred=fit)
            if self.save_vizu == True:
                plt.figure()
                plt.plot(axis_0, axis_1)
                plt.plot(axis_0, fit)
                plt.xlabel(self.line_expression)
                plt.ylabel(attribute)
                plt.title(f'R^2 = {score}, \omega = {freq:.4e}')
                plt.savefig(fname = f"{self.foldername}//R2_test_poly/{mode}_{attribute}.png")
                plt.close()

            if score > r_score_limit:
                return True, score
        return False, score    
            
    def does_data_have_same_slope_on_both_side_of_middle_point(self, attribute:str = "solid.eZZ", midpoint:float = 0, mode:int = None) -> bool:
        """The method extracts the feature (strain for example) versus the line expression (bottom-(d_top/2) for example)
        1. Extract axis0 and axis1 data.
        2. Figure out data to the left, right of the midpoint value
        3. Make a polyfit to figure out an approximate slope on each side
        4. Check if the approximate slopes have the same sign
        """
        # 1. Extract
        axis_0 = np.asarray(self.data.extract_feature(attribute=self.line_expression, eigenvalue=mode))
        axis_1 = np.asarray(self.data.extract_feature(attribute=attribute, eigenvalue=mode))
        # 2. Sort into left and right
        # data left for middle
        left_side_0 = axis_0[axis_0<midpoint]
        left_side_1 = axis_1[axis_0<midpoint] #f(x) for x > 0 --> -f(x)

        #data to the right of the middle
        right_side_0 = axis_0[axis_0>midpoint]
        right_side_1 = axis_1[axis_0>midpoint] #f(x) for x > 0 --> f(x)
        
        #3.  find coeeficient Deltay over delta x, making linear fit on both sides
        slope_right, b_right = np.polyfit(right_side_0,right_side_1, 1)
        slope_left, b_left = np.polyfit(left_side_0,left_side_1, 1)
        if self.save_vizu == True:
            plt.figure()
            plt.plot(axis_0, axis_1)
            plt.plot(left_side_0,  left_side_0*slope_left+ b_left)
            plt.plot(right_side_0,  right_side_0*slope_right+ b_right)
            plt.xlabel(self.line_expression)
            plt.ylabel(attribute)
            if np.sign(slope_left) == np.sign(slope_right): 
                plt.title(f"mode = {mode}, antisymmetric from slope approach")
            else: 
                plt.title(f"mode = {mode}, symmetric from slope approach")
            plt.savefig(fname = f"{self.foldername}/slope_test/{mode}_{attribute}.png")
            plt.close()
        # 4. Assert that the sign on the slopes are equal.
        if np.sign(slope_left) == np.sign(slope_right): 
            return True
        return False
    
    def pair_modes(self, absolute_tolerance: float = 10) -> None: #Internal function
        """ The method asserts the deflection angle and pairs flexural modes in x and y category.
        `absolute_tolerance` | `float`: refers to the absolute tolerance of how much the angles must deviate from 90 difference. Example an abs_tol = 10 means their sum
        can be 100 or 80 degrees apart. 
        """
        print("running pair modes")
        paired_with_previous_mode_by_angle = False
        paired_with_next_mode_by_angle = False
        prev_already_paired = False
        for mode in self.data.eigenvalues:
            # is the mode a flexural?
            # is 
            mode_idx = mode-1
            next_mode_idx = mode_idx+1
            prev_mode_idx = mode_idx-1
            mode_category  = self.predicted_modes[mode_idx]

            # if self.deflection_angle[next_mode_idx] > 0 and self.deflection_angle[mode_idx] < 0:
            #     pass

            # if self.deflection_angle[prev_mode_idx] < 0 and self.deflection_angle[mode_idx] > 0:
            #     pass
            

            if not mode == 1:
                paired_with_previous_mode_by_angle = math.isclose(abs(self.deflection_angle[mode_idx]) + abs(self.deflection_angle[prev_mode_idx]), 90, abs_tol=absolute_tolerance)
                if self.paired_mode[prev_mode_idx] == mode:
                    prev_already_paired = True
            if not mode == len(self.data.eigenvalues):
                paired_with_next_mode_by_angle = math.isclose(abs(self.deflection_angle[mode_idx]) + abs(self.deflection_angle[next_mode_idx]), 90, abs_tol=absolute_tolerance)
            if mode_category != "x_PureFlexural" and mode_category != "y_PureFlexural":
                self.paired_mode.append('None')
            elif prev_already_paired:
                self.paired_mode.append(mode-1)
                prev_already_paired = False
            elif paired_with_next_mode_by_angle and self.is_eigenfrequencies_close(mode, mode+1) and self.predicted_modes[next_mode_idx] in ["x_PureFlexural", "y_PureFlexural"]:
                self.paired_mode.append(mode+1)
                self.mode_pairs.append([mode,mode+1])
            elif paired_with_previous_mode_by_angle and self.is_eigenfrequencies_close(mode, mode-1) and self.predicted_modes[prev_mode_idx] in ["x_PureFlexural", "y_PureFlexural"]:
                self.paired_mode.append(mode-1)
            else:
                self.paired_mode.append('None')

            
    def is_eigenfrequencies_close(self, mode_1: int, mode_2:int, rel_tol:float = 0.1) -> bool:
        if mode_1 <1 or mode_2 < 1:
            self.passed_eigenfreq_closeness.append(False)
            return False 
        f1 = np.asarray(self.data.extract_feature(attribute='freq', eigenvalue=mode_1))[0]
        f2 = np.asarray(self.data.extract_feature(attribute='freq', eigenvalue=mode_2))[0]
        close_enough = math.isclose(f1,f2,rel_tol=rel_tol)
        self.passed_eigenfreq_closeness.append(close_enough)
        return close_enough


    def get_list_of_detected_vibrations_present(self) -> list:
        mode_list = list()
        for mode in self.data.eigenvalues:
            mode_category  = self.predicted_modes[mode-1]
            if mode_category not in mode_list:
                mode_list.append(mode_category)
        return mode_list

    def get_relative_error_compared_to_next(self) -> list:
        rel_diff_list = list()
        considered_modes = list()
        for mode in self.data.eigenvalues:
            if mode == len(self.data.eigenvalues):
                continue
            considered_modes.append(mode)
            this = self.data.extract_feature(attribute='freq', eigenvalue=mode)[0]
            next = self.data.extract_feature(attribute='freq', eigenvalue=mode+1)[0]
            rel_difference = (next-this)#/this
            rel_diff_list.append(rel_difference)
        return considered_modes, rel_diff_list
    
    def get_absolute_errors_on_angle_from_90(self):
        pair_idx = []
        abs_error_on_angle = []
        start_idx = 1
        for pair in self.mode_pairs:
            angle1 = self.deflection_angle[pair[0]-1]
            angle2 = self.deflection_angle[pair[1]-1]
            abs_error_on_angle.append(90-abs(angle1)-abs(angle2))
            pair_idx.append(start_idx)
            start_idx += 1
        return pair_idx, abs_error_on_angle
    
    def plot_absolute_error_on_angle_from_90(self, filename: str = None):
        pair_idx, abs_error_on_angle = self.get_absolute_errors_on_angle_from_90()
        fig, ax = plt.subplots(1,1)
        ax.plot(pair_idx, abs_error_on_angle, '-.')
        ax.set_xlabel('Pair number')
        ax.set_ylabel('Absolute difference from 90')
        if filename is not None:
            fig.savefig(f"{self.foldername}/{filename}.png")

    def plot_mode_num_and_freq(self):
        #for mode in self.data.eigenvalues:
        crit = np.asarray(self.predicted_modes)
        x = np.asarray(self.data.eigenvalues)
        y = np.asarray(self.frequencies)
        fig, ax = plt.subplots(1,1)
        low_lim = 0
        up_lim1 = 10
        up_lim2 = 5
        up_lim3 = 3
        up_lim4 = 1
        ax.plot(x[crit == 'x_PureFlexural'][0:up_lim1], y[crit == 'x_PureFlexural'][0:up_lim1], 'o', color = 'b')
        ax.plot(x[crit == 'y_PureFlexural'][0:up_lim1], y[crit == 'y_PureFlexural'][0:up_lim1], 'o', color = 'b')
        ax.plot(x[crit == 'PureTorsional'][0:up_lim2], y[crit == 'PureTorsional'][0:up_lim2], 'o', color = 'g')
        ax.plot(x[crit == 'PureLongitudinal'][0:up_lim3], y[crit == 'PureLongitudinal'][0:up_lim3], 'o', color = 'r')
        #ax.plot(x[crit == 'FlexuralTorsional'][0:up_lim4], y[crit == 'FlexuralTorsional'][0:up_lim4], 'o', color = 'y')
        
        #{'x_PureFlexural':[],'y_PureFlexural':[], 'PureTorsional':[], 'UnknownCategory': [],'FlexuralTorsional': [], 'PureLongitudinal': []}


    def get_pairs_relative_error_on_frequency(self):
        pair_idx = []
        rel_error_on_pairs_percentage = []
        start_idx = 1
        for pair in self.mode_pairs:
            freq1 = self.data.extract_feature(attribute='freq', eigenvalue=pair[0])[0]
            freq2 = self.data.extract_feature(attribute='freq', eigenvalue=pair[1])[0]
            rel_error_on_pairs_percentage.append((freq2 - freq1)/freq1*100)
            pair_idx.append(start_idx)
            start_idx += 1
        return pair_idx, rel_error_on_pairs_percentage

    def plot_pair_relative_error(self, filename: str = None):
        pairnum, rel_errors = self.get_pairs_relative_error_on_frequency()
        fig, axs = plt.subplots(1, 1)
        axs.plot(pairnum, rel_errors, '-.')
        axs.set_ylabel('%')
        axs.set_xlabel('Pairnumber')
        if filename is not None:
            fig.savefig(f"{self.foldername}/{filename}.png")

    def get_relative_error_compared_to_previous(self) -> list:
        rel_diff_list = list()
        considered_modes = list()
        for mode in self.data.eigenvalues:
            if mode == 1:
                continue
            considered_modes.append(mode)
            this = self.data.extract_feature(attribute='freq', eigenvalue=mode)[0]
            prev = self.data.extract_feature(attribute='freq', eigenvalue=mode-1)[0]
            rel_difference = (prev-this)#/this
            rel_diff_list.append(rel_difference)
        return considered_modes, rel_diff_list


    
    def plot_relative_error_plot_for_modes(self) -> None:
        fig, axs = plt.subplots(1, 2)

        modes, diffs = self.get_relative_error_compared_to_previous()
        axs[0].plot(modes, diffs, '.')
        modes_next, diffs_next = self.get_relative_error_compared_to_next()
        axs[1].plot(modes_next, diffs_next, '.')

        pass    

    # def take_negative_part_of_pairs(self):
    #     pass

    # def classify_mode_from_pure_strain(self, mode) -> str:
    #     pass

    def is_data_antisymmetric(self, midpoint = 0, num_error_allowed = 5/100, mode = None, attribute = 'solid.eZZ') -> bool:
        axis_0 = np.asarray(self.data.extract_feature(attribute=self.line_expression, eigenvalue=mode))
        axis_1 = np.asarray(self.data.extract_feature(attribute=attribute, eigenvalue=mode))
        integral = trapezoid(y = axis_1, x = axis_0)
        if abs(integral/max(axis_1)) < 10**(-7):
            return True
        return False

    def print_classifications_statistics(self):
        predicted_modes = np.asarray(self.predicted_modes)
        print(f"Classified Amount: {len(predicted_modes[predicted_modes != 'UnknownCategory'])}")
        print(f"Unclassified Amount: {len(predicted_modes[predicted_modes == 'UnknownCategory'])}")
        print(f"{len(predicted_modes[predicted_modes == 'PureLongitudinal'])} classified as PureLongitudinal")
        print(f"{len(predicted_modes[predicted_modes == 'FlexuralTorsional'])} classified as FlexuralTorsional")
        print(f"{len(predicted_modes[predicted_modes == 'x_PureFlexural'])} classified as x_PureFlexural")
        print(f"{len(predicted_modes[predicted_modes == 'y_PureFlexural'])} classified as y_PureFlexural")
        print(f"{len(predicted_modes[predicted_modes == 'PureTorsional'])} classified as PureTorsional")

import math as math
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    

    
# predominantly torsional

# predominnantly rotational
    # (i) not from min to max
    # (ii) not crossing 0
    # (iii) plateu around the venter

# evaluation of mixed modes
    # divergence of displacement field

# evaluate a QD at different heights! to see if difference

# evalue exx and eyy maybe
# maybe xy, yz, xz

#classi.plot_order_of_magnitude()
#print("Note that plot of magnitude agrees with method of postive, negative values.")
# So it is easy enough to peel out the 
#above -10 seems to be longitdudinal
# -8 seems to be flexural 
# lower seems to be torsional
# maybe also look at max displacement
# all displacements retracted with the mean radius of 
