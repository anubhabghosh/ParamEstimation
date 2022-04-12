function [target_pdf_log_likelihood] = compute_target_llh(x)
    target_pdf_log_likelihood = (sin(x)).^2 .* (sin(2.*x)).^2 .* mvnpdf(x);
end