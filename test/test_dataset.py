# pylint: disable=missing-docstring
import unittest

import numpy as np
import pandas as pd


STATES = ['GA','FL','fl','NY','NJ','TX']
STATUS = [1,2,3]


def _random_vals_date_range(dataset: list, high: int, date_range: pd.DatetimeIndex) -> list:
    return [
        dataset[np.random.randint(low=0,high=high)]
        for i in range(len(date_range))
    ]


def _create_dataset(count=1) -> list:
    output = []

    for _ in range(count):
        date_range = pd.date_range(start='1/1/2009', end='12/31/2012', freq='W-MON')
        output.extend(
            zip(
                _random_vals_date_range(
                    STATUS,
                    len(STATUS),
                    date_range,
                ),
                _random_vals_date_range(
                    STATES,
                    len(STATES),
                    date_range,
                ),
                np.random.randint(low=25,high=1000,size=len(date_range)),
                date_range,
            ),
        )

    return output

class TestDataset(unittest.TestCase):

    def test__create_dataset(self):
        # pylint: disable=no-self-use
        dataset = _create_dataset()
        print(dataset)
