from typing import Optional
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from fairseq import utils
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder
from fairseq.models import register_model, register_model_architecture
from fairseq.data import encoders
from torch.distributions.categorical import Categorical


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


def pad(tokens, pad_idx, eos_idx):
    tokens = tokens.clone()
    lens = []
    for i in range(tokens.shape[0]):
        eos = torch.nonzero(tokens[i] == eos_idx, as_tuple=False)
        if len(eos) > 0:
            tokens[i][eos[0]+1:] = pad_idx
            lens.append(eos[0])
        else:
            lens.append(tokens.shape[1])
    return tokens, torch.tensor(lens).cuda()



def transformer_add_parser(parser):
    parser.add_argument('--activation-fn',
                        choices=utils.get_available_activation_fns(),
                        help='activation function to use')
    parser.add_argument('--dropout', type=float, metavar='D',
                        help='dropout probability')
    parser.add_argument('--attention-dropout', type=float, metavar='D',
                        help='dropout probability for attention weights')
    parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                        help='dropout probability after activation in FFN.')
    parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                        help='path to pre-trained encoder embedding')
    parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                        help='encoder embedding dimension')
    parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                        help='encoder embedding dimension for FFN')
    parser.add_argument('--encoder-layers', type=int, metavar='N',
                        help='num encoder layers')
    parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                        help='num encoder attention heads')
    parser.add_argument('--encoder-normalize-before', action='store_true',
                        help='apply layernorm before each encoder block')
    parser.add_argument('--encoder-learned-pos', action='store_true',
                        help='use learned positional embeddings in the encoder')
    parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                        help='path to pre-trained decoder embedding')
    parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension')
    parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension for FFN')
    parser.add_argument('--decoder-layers', type=int, metavar='N',
                        help='num decoder layers')
    parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                        help='num decoder attention heads')
    parser.add_argument('--decoder-learned-pos', action='store_true',
                        help='use learned positional embeddings in the decoder')
    parser.add_argument('--decoder-normalize-before', action='store_true',
                        help='apply layernorm before each decoder block')
    parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                        help='share decoder input and output embeddings')
    parser.add_argument('--share-all-embeddings', action='store_true',
                        help='share encoder, decoder and output embeddings'
                             ' (requires shared dictionary and embed dim)')
    parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                        help='if set, disables positional embeddings (outside self attention)')
    parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                        help='comma separated list of adaptive softmax cutoff points. '
                             'Must be used with adaptive_loss criterion'),
    parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                        help='sets adaptive softmax dropout for the tail projections')
    # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
    parser.add_argument('--no-cross-attention', default=False, action='store_true',
                        help='do not perform cross-attention')
    parser.add_argument('--cross-self-attention', default=False, action='store_true',
                        help='perform cross+self-attention')
    parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                        help='perform layer-wise attention (cross-attention or cross+self-attention)')
    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                        help='LayerDrop probability for encoder')
    parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                        help='LayerDrop probability for decoder')
    parser.add_argument('--encoder-layers-to-keep', default=None,
                        help='which layers to *keep* when pruning as a comma-separated list')
    parser.add_argument('--decoder-layers-to-keep', default=None,
                        help='which layers to *keep* when pruning as a comma-separated list')
    parser.add_argument('--layernorm-embedding', action='store_true',
                        help='add layernorm to embedding')
    parser.add_argument('--no-scale-embedding', action='store_true',
                        help='if True, dont scale embeddings')
    parser.add_argument('--train-second-half', default=False,
                        action='store_true', help='otherwise train first half while freezing second half')
    parser.add_argument('--train-all', default=False,
                        action='store_true', help='train all components')
    parser.add_argument('--train-embeddings-only', default=False,
                        action='store_true', help='train embedding only')
    parser.add_argument('--pi-restore-path', default=None,
                        type=str, help='path from which to restore pi model')
    parser.add_argument('--f-restore-path', default=None,
                        type=str, help='path from which to restore base model')

    parser.add_argument('--encoder-embed-dim-2', type=int, metavar='N',
                        help='encoder embedding dimension')
    parser.add_argument('--encoder-ffn-embed-dim-2', type=int, metavar='N',
                        help='encoder embedding dimension for FFN')
    parser.add_argument('--encoder-layers-2', type=int, metavar='N',
                        help='num encoder layers')
    parser.add_argument('--encoder-attention-heads-2', type=int, metavar='N',
                        help='num encoder attention heads')
    parser.add_argument('--decoder-embed-dim-2', type=int, metavar='N',
                        help='decoder embedding dimension')
    parser.add_argument('--decoder-ffn-embed-dim-2', type=int, metavar='N',
                        help='decoder embedding dimension for FFN')
    parser.add_argument('--decoder-layers-2', type=int, metavar='N',
                        help='num decoder layers')
    parser.add_argument('--decoder-attention-heads-2', type=int, metavar='N',
                        help='num decoder attention heads')


@register_model("cls_transformer")
class CLSTransformer(TransformerModel):

    def __init__(self, args, encoder, decoder, classifier):
        super().__init__(args, encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.force_first = False
        self.padding_idx = self.encoder.padding_idx

        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        transformer_add_parser(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture_cls(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        if args.add_mask_token:
            src_dict.add_symbol('<mask>')
            if len(tgt_dict.count) != len(src_dict.count):
                tgt_dict.add_symbol('<mask>')

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        # randomly initialize the encoder
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        classifier = torch.nn.Linear(args.encoder_embed_dim, 1)

        if args.pi_restore_path is not None:
            # restore second part from checkpoint
            checkpoint = torch.load(args.pi_restore_path)
            decoder_prefix = 'decoder.'
            decoder_dict = {k[len(decoder_prefix):]: v for k, v in checkpoint['model'].items() if k.startswith(decoder_prefix)}
            decoder.load_state_dict(decoder_dict)
            encoder_prefix = 'encoder.'
            encoder_dict = {k[len(encoder_prefix):]: v for k, v in checkpoint['model'].items() if k.startswith(encoder_prefix)}
            encoder.load_state_dict(encoder_dict)

        return cls(args, encoder, decoder, classifier)

    def forward(self,
                src_tokens,
                src_lengths,
                prev_output_tokens,
                cls_input=None,
                return_all_hiddens=True,
                features_only=False,
                alignment_layer=None,
                alignment_heads=None, ):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            cls_input=cls_input,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        return decoder_out


@register_model_architecture('cls_transformer', 'cls_transformer')
def base_architecture_cls(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)


class CustomEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

    def forward_embedding(self, src_tokens, token_embedding: Optional[torch.Tensor] = None):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
        token_embedding: Optional[Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embedding)
        # # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if return_all_hiddens and len(encoder_states) > 0:
                encoder_states[-1] = x

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )



@register_model("double_transformer")
class DoubleTransformer(TransformerModel):

    def __init__(self, args, encoder_1, decoder_1, encoder_2, decoder_2, task, pi_model):
        super().__init__(args, encoder_1, decoder_1)
        self.encoder_1 = encoder_1
        self.decoder_1 = decoder_1
        self.encoder_2 = encoder_2
        self.decoder_2 = decoder_2
        self.pi_model = pi_model
        self.force_first = False
        self.train_second_half = args.train_second_half
        self.padding_idx = self.encoder_1.padding_idx
        self.train_all = args.train_all

        self.args = args
        self.task = task
        self.supports_align_args = True
        self._initialize(args)


    def _initialize(self, args):
        self.generator = self.task.build_generator(args)
        self.generator.beam_size = 1

    def decode_fn(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x

    def trainable_params(self):
        if self.train_second_half:
            # register parameters for half the model only
            # but will clear grad for whole model during train
            return chain(self.encoder_2.parameters(), self.decoder_2.parameters())
        else:
            if self.train_all:
                return self.parameters()
            else:
                return chain(self.encoder_1.parameters(), self.decoder_1.parameters())

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        transformer_add_parser(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture_double(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        if args.add_mask_token:
            src_dict.add_symbol('<mask>')
            if len(tgt_dict.count) != len(src_dict.count):
                tgt_dict.add_symbol('<mask>')

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder_1 = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder_1 = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        # update args
        old_args = {}
        for k, v in vars(args).items():
            if k.endswith('_2'):
                change_k = k[:-2]
                old_args[change_k] = getattr(args, change_k)
                setattr(args, change_k, v)

        if 'decoder_embed_dim_2' not in args:
            args.decoder_embed_dim_2 = args.decoder_embed_dim
        decoder_embed_tokens_2 = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim_2, args.decoder_embed_path
                )

        # encoder_2 = cls.build_encoder(args, tgt_dict, decoder_embed_tokens_2)
        encoder_2 = CustomEncoder(args, tgt_dict, decoder_embed_tokens_2)
        decoder_2 = cls.build_decoder(args, tgt_dict, decoder_embed_tokens_2)
        pi_model = TransformerModel(args, encoder_2, decoder_2)

        # un-update args
        for k, v in old_args.items():
            setattr(args, k, v)

        try:
            if args.pi_restore_path is not None:
                # restore second part from checkpoint
                checkpoint = torch.load(args.pi_restore_path)
                encoder_prefix = 'encoder.'
                decoder_prefix = 'decoder.'
                encoder_dict = {k[len(encoder_prefix):]: v for k, v in checkpoint['model'].items() if k.startswith(encoder_prefix)}
                decoder_dict = {k[len(decoder_prefix):]: v for k, v in checkpoint['model'].items() if k.startswith(decoder_prefix)}
                encoder_2.load_state_dict(encoder_dict)
                decoder_2.load_state_dict(decoder_dict)
        except Exception:
            raise ValueError(f"Wasn't able to load {args.pi_restore_path}; Was it moved?")

        try:
            if args.f_restore_path is not None:
                # restore first part from checkpoint
                checkpoint = torch.load(args.f_restore_path)
                encoder_prefix = 'encoder.'
                decoder_prefix = 'decoder.'
                encoder_dict = {k[len(encoder_prefix):]: v for k, v in checkpoint['model'].items() if k.startswith(encoder_prefix)}
                decoder_dict = {k[len(decoder_prefix):]: v for k, v in checkpoint['model'].items() if k.startswith(decoder_prefix)}
                # remove the encdec parameters
                encoder_1.load_state_dict(encoder_dict)
                decoder_1.load_state_dict(decoder_dict)
        except Exception:
            raise ValueError(f"Wasn't able to load {args.f_restore_path}; Was it moved?")

        return cls(args, encoder_1, decoder_1, encoder_2, decoder_2, task, pi_model)

    def _get_sample(self, src_tokens, src_lengths, prev_output_tokens, training=False, use_pi_model=False):
        # this function implicitly sets the model to eval

        # sample autoregressively from encdec, then plug back in to get lprobs
        # of the sample
        input_sample = {'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}}

        generator_model = [self]
        if use_pi_model:
            generator_model = [self.pi_model]
        sample = self.generator.generate(generator_model, input_sample)
        sample_tokens = [s[0]['tokens'] for s in sample]
        sample_tokens = pad_sequence(sample_tokens, batch_first=True, padding_value=self.padding_idx)
        # CONVERT to prev
        sample_tokens_prev = sample_tokens[sample_tokens != self.generator.eos].view(sample_tokens.shape[0], sample_tokens.shape[1] - 1)
        sample_tokens_prev = torch.cat([self.generator.eos * torch.ones((sample_tokens_prev.shape[0], 1)).long().cuda(), sample_tokens_prev], dim=1)

        sample_output = self(src_tokens, src_lengths, sample_tokens_prev, output_mode='first')
        sample_lprobs = self.get_normalized_probs(sample_output, log_probs=True)
        sample_lprobs = sample_lprobs.view(-1, sample_lprobs.size(-1))
        p_yhat = Categorical(logits=sample_lprobs)
        if training:
            return sample, sample_tokens, p_yhat, sample_tokens_prev
        else:
            return sample

    def _get_sample_simple(self, src_tokens, src_lengths, prev_output_tokens, training=False, use_pi_model=False):
        with torch.no_grad():
            # this function implicitly sets the model to eval

            # sample autoregressively from encdec, then plug back in to get lprobs
            # of the sample
            input_sample = {'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}}

            generator_model = [self]
            if use_pi_model:
                generator_model = [self.pi_model]
            sample = self.generator.generate(generator_model, input_sample)
            sample_tokens = [s[0]['tokens'] for s in sample]
            sample_tokens = pad_sequence(sample_tokens, batch_first=True, padding_value=self.padding_idx)
            # CONVERT to prev
            sample_tokens_prev = sample_tokens[sample_tokens != self.generator.eos].view(sample_tokens.shape[0], sample_tokens.shape[1] - 1)
            sample_tokens_prev = torch.cat([self.generator.eos * torch.ones((sample_tokens_prev.shape[0], 1)).long().cuda(), sample_tokens_prev], dim=1)

        sample_output = self(src_tokens, src_lengths, sample_tokens_prev, output_mode='first')
        if training:
            logits = sample_output[0]
            # mask out the padding
            padding_mask = (sample_tokens == self.padding_idx)
            pad_onehot = torch.zeros(logits.shape[2]).cuda().float()
            pad_onehot[self.padding_idx] = 1.0
            logits[padding_mask] = pad_onehot
            return sample, sample_tokens, logits
        else:
            return sample

    def forward(self,
                src_tokens,
                src_lengths,
                prev_output_tokens,
                cls_input: Optional[Tensor] = None,
                return_all_hiddens: bool = True,
                features_only: bool = False,
                alignment_layer: Optional[int] = None,
                alignment_heads: Optional[int] = None,
                output_mode : str = 'first',
                token_embedding: Optional[Tensor] = None):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        if output_mode == 'second' or self.train_second_half:
            encoder_out = self.encoder_2(
                src_tokens,
                src_lengths=src_lengths,
                cls_input=cls_input,
                return_all_hiddens=return_all_hiddens,
                token_embedding=token_embedding,
            )
            decoder_out = self.decoder_2(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            )
        elif output_mode == 'first' or self.force_first:
            encoder_out = self.encoder_1(
                src_tokens,
                src_lengths=src_lengths,
                cls_input=cls_input,
                return_all_hiddens=return_all_hiddens,
            )
            decoder_out = self.decoder_1(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            )

        return decoder_out


@register_model_architecture('double_transformer', 'double_transformer')
def base_architecture_double(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)


@register_model("custom_transformer")
class CustomTransformer(TransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.force_first = False
        self.padding_idx = self.encoder.padding_idx
        self.train_all = args.train_all

        self.args = args
        self.supports_align_args = True

    def trainable_params(self):
        if self.train_all:
            return self.parameters()
        else:
            return self.encoder.parameters()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        transformer_add_parser(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture_custom(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        if args.add_mask_token:
            src_dict.add_symbol('<mask>')
            if len(tgt_dict.count) != len(src_dict.count):
                tgt_dict.add_symbol('<mask>')

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        try:
            if args.f_restore_path is not None:
                # restore first part from checkpoint
                checkpoint = torch.load(args.f_restore_path)
                encoder_prefix = 'encoder.'
                decoder_prefix = 'decoder.'
                encoder_dict = {k[len(encoder_prefix):]: v for k, v in checkpoint['model'].items() if k.startswith(encoder_prefix)}
                decoder_dict = {k[len(decoder_prefix):]: v for k, v in checkpoint['model'].items() if k.startswith(decoder_prefix)}
                # remove the encdec parameters
                encoder.load_state_dict(encoder_dict)
                decoder.load_state_dict(decoder_dict)
        except Exception:
            raise ValueError(f"Wasn't able to load {args.f_restore_path}; Was it moved?")

        return cls(args, encoder, decoder)

    def forward(self,
                src_tokens,
                src_lengths,
                prev_output_tokens,
                cls_input: Optional[Tensor] = None,
                return_all_hiddens: bool = True,
                features_only: bool = False,
                alignment_layer: Optional[int] = None,
                alignment_heads: Optional[int] = None):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            cls_input=cls_input,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        return decoder_out


@register_model_architecture('custom_transformer', 'custom_transformer')
def base_architecture_custom(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)

