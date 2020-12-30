# pylint: disable=missing-docstring
import logging
import os
import tempfile
import unittest
import uuid
import shutil

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
                    STATES,
                    len(STATES),
                    date_range,
                ),
                _random_vals_date_range(
                    STATUS,
                    len(STATUS),
                    date_range,
                ),
                np.random.randint(low=25,high=1000,size=len(date_range)),
                date_range,
            ),
        )

    return output


class TestDataset(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level='INFO')
        super().__init__(
            *args,
            **kwargs
        )

    def test_create_dataset(self):
        # pylint: disable=no-self-use
        dataset = _create_dataset()
        temp_dir_path = os.path.join(
            tempfile.gettempdir(),
            str(uuid.uuid1()),
        )
        test_excel_filepath = os.path.join(
            temp_dir_path,
            'test.xls',
        )
        try:
            os.mkdir(temp_dir_path)
            dataframe = pd.DataFrame(
                data=dataset,
                columns=[
                    'State',
                    'Status',
                    'CustomerCount',
                    'StatusDate',
                ],
            )
            dataframe.to_excel(test_excel_filepath, index=False)
            self.assertTrue(os.path.isfile(test_excel_filepath))

            read_dataframe = pd.read_excel(
                test_excel_filepath,
                0,
            )

            states = set([])
            unique_states = read_dataframe['State'].unique()
            self.assertNotEqual(len(dataframe.index), len(unique_states))
            self.assertTrue(len(unique_states) > 0)
            for state in unique_states:
                self.assertFalse(state in states)
                states.add(state)

            states_with_status_one = read_dataframe[read_dataframe['Status'] == 1]
            self.assertTrue(len(states_with_status_one) > 0)
            for _, state in states_with_status_one.iterrows():
                self.assertTrue(state['Status'] == 1)

            state_date = read_dataframe.groupby(['State', 'StatusDate']).sum()
            self.assertTrue('Status' in state_date)
            del state_date['Status']
            self.assertFalse('Status' in state_date)
            self.assertEqual(
                'State',
                state_date.index.levels[0].name,
            )
            self.assertEqual(
                'StatusDate',
                state_date.index.levels[1].name,
            )

            state_year_month = state_date.groupby(
                [
                    state_date.index.get_level_values(0),
                    state_date.index.get_level_values(1).year,
                    state_date.index.get_level_values(1).month,
                ],
            )

            def lower_bound(datum):
                return (
                    datum.quantile(q=0.25) -
                    (1.5 * datum.quantile(q=0.75) - datum.quantile(q=0.25))
                )

            def upper_bound(datum):
                return (
                    datum.quantile(q=0.75) +
                    (1.5 * datum.quantile(q=0.75) - datum.quantile(q=0.25))
                )

            state_date['Lower'] = state_year_month['CustomerCount'].transform(lower_bound)
            state_date['Upper'] = state_year_month['CustomerCount'].transform(upper_bound)
            state_date['Outlier'] = (state_date['CustomerCount'] < state_date['Lower']) \
                | (state_date['CustomerCount'] > state_date['Upper'])
            state_date = state_date[state_date['Outlier'] == 0]
            for _, row in state_date.iterrows():
                self.assertFalse(row['Outlier'])

            daily = pd.DataFrame(
                state_date['CustomerCount'] \
                    .groupby(state_date.index.get_level_values(1)) \
                    .sum(),
            )
            daily.columns = ['CustomerCount']
            daily['Max'] = daily['CustomerCount'] \
                .groupby([lambda x: x.year, lambda x: x.month]) \
                .transform(lambda x: x.max())

            self._logger.info(daily.head())
        finally:
            if os.path.exists(temp_dir_path) and os.path.isdir(temp_dir_path):
                shutil.rmtree(temp_dir_path)
