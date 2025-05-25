use nockvm::interpreter::Context;
use nockvm::jets::util::slot;
use nockvm::jets::JetErr;
use nockvm::noun::{Cell, IndirectAtom, Noun};
use tracing::debug;
use rayon::prelude::*;

use crate::form::math::fext::*;
use crate::form::poly::Poly;
use crate::form::{Belt, FPolySlice, Felt};
use crate::hand::handle::new_handle_mut_felt;
use crate::hand::structs::HoonList;
use crate::jets::utils::jet_err;
use crate::noun::noun_ext::NounExt;

pub fn evaluate_deep_jet(context: &mut Context, subject: Noun) -> Result<Noun, JetErr> {
    let sam = slot(subject, 6)?;
    let mut sam_cur: Cell = sam.as_cell()?;

    // Extract all parameters from the subject
    let trace_evaluations = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let comp_evaluations = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let trace_elems = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let comp_elems = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let num_comp_pieces = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let weights = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let heights: Vec<u64> = HoonList::try_from(sam_cur.head())?
        .into_iter()
        .map(|x| x.as_atom().unwrap().as_u64().unwrap())
        .collect();
    sam_cur = sam_cur.tail().as_cell()?;
    let full_widths: Vec<u64> = HoonList::try_from(sam_cur.head())?
        .into_iter()
        .map(|x| x.as_atom().unwrap().as_u64().unwrap())
        .collect();
    sam_cur = sam_cur.tail().as_cell()?;
    let omega = sam_cur.head().as_felt()?;
    sam_cur = sam_cur.tail().as_cell()?;
    let index = sam_cur.head().as_atom()?.as_u64()?;
    sam_cur = sam_cur.tail().as_cell()?;
    let deep_challenge = sam_cur.head().as_felt()?;
    let new_comp_eval = sam_cur.tail().as_felt()?;

    // Convert nouns to appropriate types
    let Ok(trace_evaluations) = FPolySlice::try_from(trace_evaluations) else {
        debug!("trace_evaluations is not a valid FPolySlice");
        return jet_err();
    };
    let Ok(comp_evaluations) = FPolySlice::try_from(comp_evaluations) else {
        debug!("comp_evaluations is not a valid FPolySlice");
        return jet_err();
    };
    let trace_elems: Vec<Belt> = HoonList::try_from(trace_elems)?
        .into_iter()
        .map(|x| x.as_atom().unwrap().as_u64().unwrap())
        .map(|x| Belt(x))
        .collect();
    let comp_elems: Vec<Belt> = HoonList::try_from(comp_elems)?
        .into_iter()
        .map(|x| x.as_atom().unwrap().as_u64().unwrap())
        .map(|x| Belt(x))
        .collect();
    let num_comp_pieces = num_comp_pieces.as_atom()?.as_u64()?;
    let Ok(weights) = FPolySlice::try_from(weights) else {
        debug!("weights is not a valid FPolySlice");
        return jet_err();
    };
    // let heights: Vec<u64> = HoonList::try_from(heights)?
    //     .into_iter()
    //     .map(|x| x.as_atom().unwrap().as_u64().unwrap())
    //     .collect();
    // let full_widths: Vec<u64> = HoonList::try_from(full_widths)?
    //     .into_iter()
    //     .map(|x| x.as_atom().unwrap().as_u64().unwrap())
    //     .collect();
    // let omega = omega.as_felt()?;
    // let index = index.as_atom()?.as_u64()?;
    // let deep_challenge = deep_challenge.as_felt()?;
    // let new_comp_eval = new_comp_eval.as_felt()?;

    //  TODO use g defined wherever it is
    let g = Felt::lift(Belt(7));
    let omega_pow = fmul_(&fpow_(&omega, index as u64), &g);

    // Parallelized loops for processing trace columns
    let (acc, num) = heights.par_iter().enumerate().try_reduce(
        || Ok((Felt::zero(), 0usize)), // Initial value: (accumulator, element_index)
        |acc_num_res, (i, &height)| -> Result<(Felt, usize), JetErr> {
            let (mut current_acc, mut current_num) = acc_num_res?;
            let full_width = full_widths[i] as usize;
            let omicron = Felt::lift(Belt(height).ordered_root()?);

            let current_trace_elems = &trace_elems[current_num..(current_num + full_width)];

            // Process first row trace columns
            let denom = fsub_(&omega_pow, &deep_challenge);
            (current_acc, _) = process_belt(
                current_trace_elems, &trace_evaluations.0, &weights.0, full_width, 0, &denom, &current_acc,
            );

            // Process second row trace columns (shifted by omicron)
            let denom = fsub_(&omega_pow, &fmul_(&deep_challenge, &omicron));
            (current_acc, _) = process_belt(
                current_trace_elems, &trace_evaluations.0, &weights.0, full_width, 0, &denom, &current_acc,
            );

            current_num += full_width;
            Ok((current_acc, current_num))
        },
    )?;

    // Process composition elements (this part is not parallelized as it's a single iteration)
    let denom = fsub_(&omega_pow, &fpow_(&deep_challenge, num_comp_pieces as u64));
    let (acc, num) = process_felt(
        &comp_elems, &comp_evaluations.0, &weights.0, num_comp_pieces as usize, num, &denom, &acc,
    )?;

    let (res_atom, res_felt): (IndirectAtom, &mut Felt) = new_handle_mut_felt(&mut context.stack);
    *res_felt = finv_(&acc)?; // Invert the accumulated value

    assert!(felt_atom_is_valid(res_atom));
    Ok(res_atom.as_noun())
}

#[inline(always)]
fn process_belt(
    elems: &[Belt],
    evaluations: &[Felt],
    weights: &[Felt],
    full_width: usize,
    mut num: usize,
    denom: &Felt,
    acc: &Felt,
) -> (Felt, usize) {
    let mut current_acc = *acc;
    for j in 0..full_width {
        let num_j = Felt::lift(elems[j]);
        let term = fmul_(&weights[num], &fmul_(&num_j, &evaluations[num]));
        let term = fdiv_(&term, &denom);
        current_acc = fadd_(&current_acc, &term);
        num += 1;
    }
    (current_acc, num)
}

#[inline(always)]
fn process_felt(
    elems: &[Belt],
    evaluations: &[Felt],
    weights: &[Felt],
    full_width: usize,
    mut num: usize,
    denom: &Felt,
    acc: &Felt,
) -> Result<(Felt, usize), JetErr> {
    let mut current_acc = *acc;
    for j in 0..full_width {
        let num_j = Felt::lift(elems[j]);
        let term = fmul_(&weights[num], &fmul_(&num_j, &evaluations[num]));
        let term = fdiv_(&term, &denom);
        current_acc = fadd_(&current_acc, &term);
        num += 1;
    }
    Ok((current_acc, num))
}

#[inline(always)]
fn fadd_(a: &Felt, b: &Felt) -> Felt {
    a + b
}

#[inline(always)]
fn fsub_(a: &Felt, b: &Felt) -> Felt {
    a - b
}

#[inline(always)]
fn fmul_(a: &Felt, b: &Felt) -> Felt {
    a * b
}

#[inline(always)]
fn fpow_(a: &Felt, n: u64) -> Felt {
    a.pow(n)
}

#[inline(always)]
fn finv_(a: &Felt) -> Result<Felt, JetErr> {
    a.inverse().map_err(|_| jet_err())
}

#[inline(always)]
fn fdiv_(a: &Felt, b: &Felt) -> Result<Felt, JetErr> {
    a.div(b).map_err(|_| jet_err())
}

//  Return true if noun can be converted to Felt
#[inline(always)]
fn felt_atom_is_valid(noun: IndirectAtom) -> bool {
    // Felt is U256
    noun.size() == 4 // four u64's
}
