# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import array_api_compat

from mdescriptor.descriptors.model_backed._vendor.dpa4desc.dpmodel.array_api import (
    Array,
)
from mdescriptor.descriptors.model_backed._vendor.dpa4desc.dpmodel.common import (
    NativeOP,
)
from mdescriptor.descriptors.model_backed._vendor.dpa4desc.dpmodel.utils.network import (
    EmbeddingNet,
)
from mdescriptor.descriptors.model_backed._vendor.dpa4desc.utils.version import (
    check_version_compatibility,
)


def _array_device_or_none(array: Array) -> Any:
    try:
        return array_api_compat.device(array)
    except AttributeError:
        return None


def remap_atype_to_padding(atype: Array, ntypes_with_padding: int) -> Array:
    """Map negative placeholder types to a padded table's final row.

    Parameters
    ----------
    atype : Array
        Atom-type indices. Negative entries denote virtual or padding atoms.
    ntypes_with_padding : int
        Number of rows in a table that reserves its final row for padding.

    Returns
    -------
    Array
        Atom-type indices with every negative entry replaced by
        ``ntypes_with_padding - 1``.

    Notes
    -----
    This sentinel convention is valid only for tables that explicitly include
    a final padding row, such as descriptor type-embedding and type-pair
    tables. It must not be used for real-type-only tables such as ``davg``,
    ``dstd``, or spin masks; virtual entries must be masked or clamped to a
    valid real type before indexing those tables.
    """
    xp = array_api_compat.array_namespace(atype)
    return xp.where(
        atype >= 0,
        atype,
        xp.full_like(atype, ntypes_with_padding - 1),
    )


def take_type_embedding(type_embedding: Array, atype: Array) -> Array:
    """Gather type embeddings, mapping virtual atom types to the padding row.

    Parameters
    ----------
    type_embedding : Array
        Type-embedding table whose final row is reserved for virtual or
        padding atoms.
    atype : Array
        Atom-type indices with arbitrary shape. Negative entries denote
        virtual or padding atoms.

    Returns
    -------
    Array
        Gathered embeddings with shape ``(*atype.shape,
        type_embedding.shape[-1])``.

    Notes
    -----
    ``TypeEmbedNet`` reconstructs a literal zero padding row on every call.
    ``SeZMTypeEmbedding`` stores its reserved row in the trainable embedding
    array and initializes it to zero. This helper guarantees selection of the
    reserved row; the table implementation remains responsible for keeping
    that row neutral.

    Negative placeholder types must be remapped explicitly because negative
    gather indices either wrap or fail depending on the array backend.
    """
    # The caller's atom-type array determines the active backend. Model
    # conversion keeps the embedding table in that same namespace while
    # preserving trainable tensors and their gradients.
    xp = array_api_compat.array_namespace(atype)
    safe_atype = remap_atype_to_padding(atype, type_embedding.shape[0])
    return xp.take(type_embedding, xp.astype(safe_atype, xp.int64), axis=0)


class TypeEmbedNet(NativeOP):
    r"""Type embedding network.

    Each atom type :math:`t` is represented by a one-hot vector
    :math:`\mathbf e_t` (or an electronic-configuration vector), then mapped
    by an embedding network :math:`\mathcal N`:

    .. math::

       \mathbf T_t=\mathcal N(\mathbf e_t).

    If ``padding`` is enabled, an additional all-zero row represents padded
    neighbor-list entries.

    Parameters
    ----------
    ntypes : int
        Number of atom types
    neuron : list[int]
        Number of neurons in each hidden layers of the embedding net
    resnet_dt
        Time-step `dt` in the resnet construction: y = x + dt * \phi (Wx + b)
    activation_function
        The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision
        The precision of the embedding net parameters. Supported options are |PRECISION|
    trainable
        If the weights of embedding net are trainable.
    seed
        Random seed for initializing the network parameters.
    padding
        Concat the zero padding to the output, as the default embedding of empty type.
    use_econf_tebd: bool, Optional
        Whether to use electronic configuration type embedding.
    use_tebd_bias : bool, Optional
        Whether to use bias in the type embedding layer.
    type_map: list[str], Optional
        A list of strings. Give the name to each type of atoms.
    """

    def __init__(
        self,
        *,
        ntypes: int,
        neuron: list[int],
        resnet_dt: bool = False,
        activation_function: str = "tanh",
        precision: str = "default",
        trainable: bool = True,
        seed: int | list[int] | None = None,
        padding: bool = False,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: list[str] | None = None,
    ) -> None:
        self.ntypes = ntypes
        self.neuron = neuron
        self.seed = seed
        self.resnet_dt = resnet_dt
        self.precision = precision
        self.activation_function = str(activation_function)
        self.trainable = trainable
        self.padding = padding
        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.type_map = type_map
        embed_input_dim = ntypes
        if self.use_econf_tebd:
            raise NotImplementedError(
                "Electronic-configuration type embedding (use_econf_tebd=True) "
                "is not supported by mdescriptor.descriptors.model_backed._vendor.dpa4desc."
            )
        self.embedding_net = EmbeddingNet(
            embed_input_dim,
            self.neuron,
            self.activation_function,
            self.resnet_dt,
            self.precision,
            seed=self.seed,
            bias=self.use_tebd_bias,
            trainable=trainable,
        )

    def call(self) -> Array:
        r"""Return all type embeddings :math:`\mathbf T_t=\mathcal N(\mathbf e_t)`."""
        sample_array = self.embedding_net[0]["w"]
        xp = array_api_compat.array_namespace(sample_array)
        if not self.use_econf_tebd:
            embed = self.embedding_net(
                xp.eye(
                    self.ntypes,
                    dtype=sample_array.dtype,
                    device=_array_device_or_none(sample_array),
                )
            )
        else:
            raise NotImplementedError(
                "Electronic-configuration type embedding (use_econf_tebd=True) "
                "is not supported by mdescriptor.descriptors.model_backed._vendor.dpa4desc."
            )
        if self.padding:
            embed_pad = xp.zeros(
                (1, embed.shape[-1]),
                dtype=embed.dtype,
                device=_array_device_or_none(embed),
            )
            embed = xp.concat([embed, embed_pad], axis=0)
        return embed

    @classmethod
    def deserialize(cls, data: dict) -> "TypeEmbedNet":
        """Deserialize the model.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        Model
            The deserialized model
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data_cls = data.pop("@class")
        assert data_cls == "TypeEmbedNet", f"Invalid class {data_cls}"

        embedding_net = EmbeddingNet.deserialize(data.pop("embedding"))
        # compat with version 1
        if "use_tebd_bias" not in data:
            data["use_tebd_bias"] = True
        type_embedding_net = cls(**data)
        type_embedding_net.embedding_net = embedding_net
        return type_embedding_net

    def serialize(self) -> dict:
        """Serialize the model.

        Returns
        -------
        dict
            The serialized data
        """
        return {
            "@class": "TypeEmbedNet",
            "@version": 2,
            "ntypes": self.ntypes,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "precision": self.precision,
            "activation_function": self.activation_function,
            "trainable": self.trainable,
            "padding": self.padding,
            "use_econf_tebd": self.use_econf_tebd,
            "use_tebd_bias": self.use_tebd_bias,
            "type_map": self.type_map,
            "embedding": self.embedding_net.serialize(),
        }
