"""
This file provides an minor adaptation to the sklearn OneVsRestClassifier that can work with a list of differently configured estimators.
The standard implementation of OneVsRestClassifier uses a single estimator and copies it for every class.
Metadata routing has also been adapted, so different metadata can be routed to each estimator.

note: last used with sklearn 1.4.2
"""

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping
from sklearn.multiclass import _fit_binary, process_routing, _raise_for_params


class FlexOneVsRestClassifier(OneVsRestClassifier):
    """
    OneVsRestClassifier that works on a list of estimators.
    Note that the order of estimators is important, the estimator at index i will be applied to the target variable in column i in the fit() method.
    Metadata can be routed to each of the estimators independently.
    Use .set_*_request() on the individual estimators before passing to request metadata.
    """

    def __init__(self, estimators, n_jobs=1):
        self.estimators = []
        for estimator in estimators:
            self.add_estimator(estimator)
        self.n_jobs = n_jobs


    def add_estimator(self, estimator):
        self.estimators.append(estimator)


    def fit(self, X, y, **fit_params):

        _raise_for_params(fit_params, self, "fit")
        routed_params = process_routing(
            self,
            "fit",
            **fit_params,
        )

        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = [col.toarray().ravel() for col in Y.T]

        if len(columns) != len(self.estimators):
            raise ValueError("Number of target classes must match number of estimators.")

        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_binary)(
                self.estimators[i],
                X,
                column,
                fit_params=routed_params[f"estimator_{i}"].fit,
                classes=[
                    "not %s" % self.label_binarizer_.classes_[i],
                    self.label_binarizer_.classes_[i],
                ],
            )
            for i, column in enumerate(columns)
        )
        return self


    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = (
            MetadataRouter(owner=self.__class__.__name__)
        )
        for i in range(len(self.estimators)):
            est = {f"estimator_{i}" : self.estimators[i]}
            router.add(
                **est,
                method_mapping=MethodMapping()
                .add(callee="fit", caller="fit")
                .add(callee="partial_fit", caller="partial_fit"),
            )
        return router