from typing import Optional
import uuid

from sqlalchemy import ARRAY, BigInteger, Double, Identity, Integer, PrimaryKeyConstraint, Text, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass


class VisDialCLIPAnswers(Base):
    __tablename__ = 'VisDialCLIPAnswers'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='VisDialCLIPAnswers_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    idx: Mapped[Optional[int]] = mapped_column(BigInteger)
    answers: Mapped[Optional[str]] = mapped_column(Text)
    ans_em: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Double(precision=53)))
    mode: Mapped[Optional[str]] = mapped_column(Text)


class VisDialCLIPCapDial(Base):
    __tablename__ = 'VisDialCLIPCapDial'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='VisDialCLIPCapDial_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    image_id: Mapped[Optional[str]] = mapped_column(Text)
    caption: Mapped[Optional[str]] = mapped_column(Text)
    dialog_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    img_em: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Double(precision=53)))
    cap_em: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Double(precision=53)))
    mode: Mapped[Optional[str]] = mapped_column(Text)
    image_path: Mapped[Optional[str]] = mapped_column(Text)


class VisDialCLIPDial(Base):
    __tablename__ = 'VisDialCLIPDial'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='VisDialCLIPDial_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    dialog_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    answer: Mapped[Optional[int]] = mapped_column(Integer)
    question: Mapped[Optional[int]] = mapped_column(Integer)
    answer_options: Mapped[Optional[list[int]]] = mapped_column(ARRAY(Integer()))
    mode: Mapped[Optional[str]] = mapped_column(Text)


class VisDialCLIPQuestions(Base):
    __tablename__ = 'VisDialCLIPQuestions'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='VisDialCLIPQuestions_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    idx: Mapped[Optional[int]] = mapped_column(BigInteger)
    question: Mapped[Optional[str]] = mapped_column(Text)
    q_em: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Double(precision=53)))
    mode: Mapped[Optional[str]] = mapped_column(Text)
