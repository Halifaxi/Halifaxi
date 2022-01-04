from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt


class PredictionData:
    """Encapuslates file parsing, computations, and plotting.

    Takes in a csv file from the World Happiness Report and can standardize
    along with create a regression estimate from the data using least
    squares. Contains methods such as standardize(), and lstsq_model().

    Args:
        file : Takes a csv file of the World Happiness Report data. 

    Attributes:
        file : Takes in the world happiness report csv data.
        data : Loads all of the data into a numpy array for further manipulation.
        score : A rating of the subjective happiness of the country.
        gdp_pc : A countries GDP per capita.
        social_support : Assessment of a countries social support.
        life_exp : Life expectancy within a country.
        freedom : Assessments of freedom within a country.
        generosity : How generous the people are within a country.
        corruption: Perceptions of corruption within a country. 

    """

    def __init__(self, file: str) -> None:
        self.file: str = file
        self.data: np.ndarray = np.loadtxt(file,
                                           delimiter=',',
                                           usecols=(range(2, 9)),
                                           skiprows=1)
        self.score = self.data.T[0]
        self.gdp_pc = self.data.T[1]
        self.social_support = self.data.T[2]
        self.life_exp = self.data.T[3]
        self.freedom = self.data.T[4]
        self.generosity = self.data.T[5]
        self.corruption = self.data.T[6]

    def standardize(self) -> None:
        """For all features, sets the mean = 0 and sd = 1.
        """
        def calc(x): return (x - np.mean(x)) / x.std()
        self.score = calc(self.score)
        self.gdp_pc = calc(self.gdp_pc)
        self.life_exp = calc(self.life_exp)
        self.social_support = calc(self.social_support)
        self.freedom = calc(self.freedom)
        self.generosity = calc(self.generosity)
        self.corruption = calc(self.corruption)

    def lstsq_model(self, feature: np.ndarray) -> Tuple[Dict, int]:
        """Calculates the least squares fit for a given feature.

        Args:
            feature : A feature to use as the independent variable.

        Returns:
            A tuple with the first entry being a dictionary with key equal to 
                the beta and value equal to the beta value. The second entry is
                the sum of squared residual value. 

        """
        A = np.append(np.ones((len(feature), 1)),
                      feature.reshape(len(feature), 1), axis=1)
        betas, Resid, _, ew_ = np.linalg.lstsq(A, self.score, rcond=None)
        y_hat = betas[0] + betas[1] * feature
        resid = np.sum((self.score-y_hat)**2)
        return ({'B0': betas[0], 'B1': betas[1]}, resid)

    def find_best_feature(self, predictor: np.ndarray) -> np.ndarray:
        """ Creates a y_hat estimate from a given feature.

        Args:
            predictor : Any given feature from the happiness report.

        Returns:
            The y_hat estimate from that predictor.

        """
        betas, R = self.lstsq_model(predictor)
        print(f'The Sum of Squared Residuals for Social Support is {R}')
        return betas['B0'] + betas['B1'] * predictor
