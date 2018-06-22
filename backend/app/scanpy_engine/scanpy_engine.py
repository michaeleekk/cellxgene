import scanpy.api as sc
import numpy as np
import os

from ..util.schema_parse import parse_schema
from ..driver.driver import CXGDriver


class ScanpyEngine(CXGDriver):

	def __init__(self, data, schema=None, graph_method="umap", diffexp_method="ttest"):
		self.data = self._load_data(data)
		self.schema = self._load_or_infer_schema(data, schema)
		self._set_cell_ids()
		self.cell_count = self.data.shape[0]
		# TODO Do I need this?
		self.gene_count = self.data.shape[1]
		self.graph_method = graph_method
		self.diffexp_method = diffexp_method


	@staticmethod
	def _load_data(data):
		return sc.read(os.path.join(data, "data.h5ad"))

	@staticmethod
	def _load_or_infer_schema(data, schema):
		data_schema = None
		if not schema:
			pass
		else:
			data_schema = parse_schema(os.path.join(data,schema))
		return data_schema

	def _set_cell_ids(self):
		self.data.obs['cxg_cell_id'] = list(range(self.data.obs.shape[0]))
		self.data.obs["cell_name"] = list(self.data.obs.index)
		self.data.obs.set_index('cxg_cell_id', inplace=True)

	def cells(self):
		return list(self.data.obs.index)

	def cellids(self, df=None):
		if df:
			return list(df.obs.index)
		else:
			return list(self.data.obs.index)

	def genes(self):
		return self.data.var.index.tolist()

	def filter_cells(self, filter):
		"""
		Filter cells from data and return a subset of the data
		:param filter:
		:return: iterator through cell ids
		"""
		cell_idx = np.ones((self.cell_count,), dtype=bool)
		for key, value in filter.items():
			if value["variable_type"] == "categorical":
				key_idx = np.in1d(getattr(self.data.obs, key), value["query"])
				cell_idx = np.logical_and(cell_idx, key_idx)
			else:
				min_ = value["query"]["min"]
				max_ = value["query"]["max"]
				if min_:
					key_idx = np.array((getattr(self.data.obs, key) >= min_).data)
					cell_idx = np.logical_and(cell_idx, key_idx)
				if max_:
					key_idx = np.array((getattr(self.data.obs, key) <= min_).data)
					cell_idx = np.logical_and(cell_idx, key_idx)
		return self.data[cell_idx, :]

	def metadata_ranges(self, df=None):
		metadata_ranges = {}
		if not df:
			df = self.data
		for field in self.schema:
			if self.schema[field]["variabletype"] == "categorical":
				group_by = field
				if group_by == "CellName":
					group_by = 'cell_name'
				metadata_ranges[field] = {"options": df.obs.groupby(group_by).size().to_dict()}
			else:
				metadata_ranges[field] = {
					"range": {
						"min": df.obs[field].min(),
						"max": df.obs[field].max()
					}
				}
		return metadata_ranges

	def metadata(self, df, fields=None):
		"""
		Generator for metadata. Gets the metadata values cell by cell and returns all value
		or only certain values if names is not None

		"""
		metadata = df.obs.to_dict(orient="records")
		for idx in range(len(metadata)):
			metadata[idx]["CellName"] = metadata[idx].pop("cell_name", None)
		return metadata


	def create_graph(self, df):
		"""
		Computes a n-d layout for cells through dimensionality reduction.
		"""
		getattr(sc.tl, self.graph_method)(df)
		graph = df.obsm["X_{graph_method}".format(graph_method=self.graph_method)]
		normalized_graph = (graph - graph.min()) / (graph.max() - graph.min())
		return np.hstack((df.obs["cell_name"].values.reshape(len(df.obs.index), 1), normalized_graph)).tolist()


	def diffexp(self, cells_iterator_1, cells_iterator_2):
		pass

	def expression(self, ):
		pass






