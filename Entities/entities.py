from typing import Optional
import datetime
import uuid

from sqlalchemy import ARRAY, BigInteger, Boolean, Date, DateTime, Double, Identity, Integer, PrimaryKeyConstraint, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass


class IRESGMSCOCOV2(Base):
    __tablename__ = 'IRESG_MSCOCO_V2'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='IRESG_MSCOCO_V2_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    image_id: Mapped[Optional[str]] = mapped_column(Text)
    embedding: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Double(precision=53)))
    triplets: Mapped[Optional[str]] = mapped_column(Text)


class IRESGVGV2(Base):
    __tablename__ = 'IRESG_VG_V2'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='IRESG_VG_V2_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    image_id: Mapped[Optional[str]] = mapped_column(Text)
    embedding: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Double(precision=53)))
    triplets: Mapped[Optional[str]] = mapped_column(Text)


class Pages(Base):
    __tablename__ = 'Pages'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='Pages_pkey'),
    )

    ID: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    PageName: Mapped[Optional[str]] = mapped_column(Text)
    PageURL: Mapped[Optional[str]] = mapped_column(Text)
    PageLogo: Mapped[Optional[str]] = mapped_column(Text)
    RoleID: Mapped[Optional[list[int]]] = mapped_column(ARRAY(Integer()))
    Activate: Mapped[Optional[bool]] = mapped_column(Boolean)
    Delete: Mapped[Optional[bool]] = mapped_column(Boolean)
    CreateAt: Mapped[Optional[datetime.date]] = mapped_column(Date)
    UpdateAt: Mapped[Optional[datetime.date]] = mapped_column(Date)
    DeleteAt: Mapped[Optional[datetime.date]] = mapped_column(Date)


class Role(Base):
    __tablename__ = 'Role'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='Role_pkey'),
    )

    ID: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    RoleName: Mapped[Optional[str]] = mapped_column(Text)
    CreateAt: Mapped[Optional[datetime.date]] = mapped_column(Date)
    UpdateAt: Mapped[Optional[datetime.date]] = mapped_column(Date)
    Active: Mapped[Optional[bool]] = mapped_column(Boolean)


class TestTable(Base):
    __tablename__ = 'TestTable'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='TestTable_pkey'),
    )

    ID: Mapped[int] = mapped_column(Integer, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=2147483647, cycle=False, cache=1), primary_key=True)
    TestContent1: Mapped[Optional[str]] = mapped_column(Text)
    TestContent2: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Double(precision=53)))


class User(Base):
    __tablename__ = 'User'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='User_pkey'),
    )

    ID: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    Username: Mapped[Optional[str]] = mapped_column(Text)
    Password: Mapped[Optional[str]] = mapped_column(Text)
    RoleID: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text()))
    Fullname: Mapped[Optional[str]] = mapped_column(Text)
    CreateAt: Mapped[Optional[datetime.date]] = mapped_column(Date)
    UpdateAt: Mapped[Optional[datetime.date]] = mapped_column(Date)
    Activate: Mapped[Optional[bool]] = mapped_column(Boolean)
    Delete: Mapped[Optional[bool]] = mapped_column(Boolean)
    DeleteAt: Mapped[Optional[datetime.date]] = mapped_column(Date)


class VisDialCLIPAnswers(Base):
    __tablename__ = 'VisDialCLIPAnswers'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='VisDialCLIPAnswers_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
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
    dialog_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
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
    question: Mapped[Optional[str]] = mapped_column(Text)
    q_em: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Double(precision=53)))
    mode: Mapped[Optional[str]] = mapped_column(Text)


class VisDialTargetAnnotations(Base):
    __tablename__ = 'VisDialTargetAnnotations'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='VisDialTargetAnnotations_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    split: Mapped[Optional[str]] = mapped_column(Text)
    dialog_index: Mapped[Optional[int]] = mapped_column(BigInteger)
    image_id: Mapped[Optional[str]] = mapped_column(Text)
    image_path: Mapped[Optional[str]] = mapped_column(Text)
    base_caption: Mapped[Optional[str]] = mapped_column(Text)
    dialogue: Mapped[Optional[dict]] = mapped_column(JSONB)
    visual_facts: Mapped[Optional[dict]] = mapped_column(JSONB)
    positive_facts: Mapped[Optional[dict]] = mapped_column(JSONB)
    negative_facts: Mapped[Optional[dict]] = mapped_column(JSONB)
    uncertain_facts: Mapped[Optional[dict]] = mapped_column(JSONB)
    enriched_caption: Mapped[Optional[str]] = mapped_column(Text)
    source: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True))
