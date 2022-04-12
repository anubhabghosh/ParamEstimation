function [llh_pdf] = proposal_pdf_unif_log_likelihood(y, x, alpha)
    llh_pdf = unifpdf(y, x - alpha, x + alpha);
end
