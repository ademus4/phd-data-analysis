import luigi
import os
from glob import glob

import pipeline


class ProcessData(luigi.WrapperTask):
    input_path = luigi.Parameter()
    folder = luigi.Parameter(default="data")
    output_dir = luigi.Parameter(pipeline.DefaultParams().output_dir)

    def requires(self):
        # get a list of hipo files in the path and submit them as plotting tasks
        input_files = glob(self.input_path)
        output_path = os.path.join(self.output_dir, self.folder)
        for item in input_files:
            yield pipeline.ApplyCuts(item, output_dir=output_path)


class AnalyseMoments(luigi.WrapperTask):
    data_path = luigi.Parameter()
    sims_path = luigi.Parameter()

    def requires(self):
        # glob for correct files in data/sims dirs
        # give these files to moments task
        yield


class MCMCPlotsBins(luigi.WrapperTask):
    input_dir = luigi.Parameter(
        os.path.join(pipeline.DefaultParams().output_dir, "moments"))

    def requires(self):
        files = glob(os.path.join(self.input_dir, "**", "ResultsHS*.root"))
        for item in files:
            yield pipeline.MCMCMomentsPlotsPerBin(input_file=item)


if __name__ == '__main__':
    luigi.build(
        [ProcessData()],
        workers=1, 
        local_scheduler=True
    )
